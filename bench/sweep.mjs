// Optimization sweep. Measures generation throughput, prefill as context
// grows, concurrent serving across parallel slots, KV cache cost at long
// context, and prefix cache reuse.
//
//   node bench/sweep.mjs
//
// Output: bench/results.json plus a markdown table on stdout.
import { spawn, spawnSync } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { performance } from 'node:perf_hooks';
import fs from 'node:fs';

const LLAMA = process.env.LLAMA_SERVER || `${process.env.HOME}/llama.cpp/build/bin/llama-server`;
const MODEL_DIR = process.env.MODEL_DIR || `${process.env.HOME}/models`;
const MODEL = process.env.SWEEP_MODEL || 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf';
const ALIAS = process.env.SWEEP_ALIAS || 'sweep';
const PORT = Number(process.env.SWEEP_PORT || 8099);

const SHORT = 'Explain what a hash table is in three sentences.';
// ~1500 tokens of filler, to push prefill.
const LONG = 'The following is a system log excerpt. ' + 'Request accepted, worker dispatched, response streamed back to the client over the wire. '.repeat(60) + '\n\nSummarize the log in one sentence.';

function vramUsed() {
  const out = spawnSync('rocm-smi', ['--showmeminfo', 'vram'], { encoding: 'utf8' }).stdout || '';
  const m = out.match(/Total Used Memory \(B\):\s*(\d+)/);
  return m ? Number(m[1]) : null;
}

function startServer(cfg) {
  const args = [
    '-m', `${MODEL_DIR}/${MODEL}`,
    '-a', ALIAS,
    '-c', String(cfg.ctx),
    '-ngl', String(cfg.ngl),
    '-t', String(cfg.threads ?? 8),
    '-np', String(cfg.np ?? 1),
    '--host', '127.0.0.1',
    '--port', String(PORT),
    '--jinja',
    '-fa', cfg.fa ?? 'on',
    '-ctk', cfg.kv,
    '-ctv', cfg.kv,
    '--reasoning', 'off',
  ];
  if (cfg.cacheReuse) args.push('--cache-reuse', String(cfg.cacheReuse));
  return spawn(LLAMA, args, { stdio: ['ignore', 'ignore', 'ignore'] });
}

async function waitForServer(timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/v1/models`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return true;
    } catch {}
    await sleep(1500);
  }
  return false;
}

async function chat(prompt, maxTokens) {
  const started = performance.now();
  const res = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: ALIAS,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: maxTokens,
      temperature: 0.7,
    }),
  });
  const json = await res.json();
  const ms = performance.now() - started;
  const usage = json.usage || {};
  return {
    ms,
    prompt: usage.prompt_tokens || 0,
    gen: usage.completion_tokens || 0,
    tps: usage.completion_tokens ? (usage.completion_tokens / ms) * 1000 : 0,
    ptps: usage.prompt_tokens ? (usage.prompt_tokens / ms) * 1000 : 0,
  };
}

async function measure(cfg, label, fn) {
  process.stdout.write(`${label} ... `);
  const child = startServer(cfg);
  const ready = await waitForServer();
  if (!ready) {
    console.log('FAILED TO START');
    child.kill('SIGKILL');
    await sleep(2000);
    return null;
  }
  const vram = vramUsed();
  const value = await fn();
  const vramAfter = vramUsed();
  console.log(`${value.label_text} | vram ${vramAfter ? (vramAfter / 1e9).toFixed(1) : '?'} GB`);
  const row = { ...cfg, label, ...value, vram_bytes: vramAfter, vram_loaded_bytes: vram };
  child.kill('SIGKILL');
  await sleep(2500);
  return row;
}

const results = [];

// 1. Generation throughput, longer output so the measurement is not dominated
//    by first-token overhead.
results.push(await measure({ ctx: 4096, ngl: 99, kv: 'q8_0', fa: 'on' }, 'gen 4096 ctx', async () => {
  const runs = [];
  for (let i = 0; i < 3; i++) runs.push(await chat(SHORT, 256));
  const tps = runs.reduce((s, r) => s + r.tps, 0) / runs.length;
  const ms = runs.reduce((s, r) => s + r.ms, 0) / runs.length;
  return { gen_tps: Number(tps.toFixed(1)), gen_ms: Math.round(ms), gen_tokens: runs[0].gen, label_text: `${tps.toFixed(1)} tok/s, ${Math.round(ms)} ms` };
}));

// 2. Prefill: a long prompt, so the context window is the variable.
for (const ctx of [2048, 4096, 8192]) {
  results.push(await measure({ ctx, ngl: 99, kv: 'q8_0', fa: 'on' }, `prefill ctx ${ctx}`, async () => {
    const r = await chat(LONG, 16);
    return { prefill_tps: Number(r.ptps.toFixed(1)), prefill_prompt_tokens: r.prompt, label_text: `${r.prompt} prompt tok in ${Math.round(r.ms)} ms` };
  }));
}

// 3. Concurrent serving. Parallel slots let the server batch requests.
//    Each slot gets a 4096-token window, so ctx scales with np. np=8/16 are
//    the interesting points: batching keeps weight reads shared across slots,
//    and the aggregate keeps climbing well past np=4.
for (const np of [1, 2, 4, 8, 16]) {
  results.push(await measure({ ctx: np * 4096, ngl: 99, kv: 'q8_0', fa: 'on', np }, `concurrent np=${np}`, async () => {
    const t0 = performance.now();
    const runs = await Promise.all(Array.from({ length: np }, () => chat(SHORT, 256)));
    const wall = performance.now() - t0;
    const totalGen = runs.reduce((s, r) => s + r.gen, 0);
    return { concurrent_tps: Number(((totalGen / wall) * 1000).toFixed(1)), concurrent_wall_ms: Math.round(wall), label_text: `${((totalGen / wall) * 1000).toFixed(1)} tok/s aggregate, ${Math.round(wall)} ms wall` };
  }));
}

// 4. KV cache cost. At 8192 the difference between f16 and q4_0 shows up in VRAM.
//    f16 is also the fastest cache type: the KV read is on the critical path of
//    flash attention, and the quantize/dequantize work costs more than the
//    bytes it saves at this size.
for (const kv of ['f16', 'q8_0', 'q4_0']) {
  results.push(await measure({ ctx: 4096, ngl: 99, kv, fa: 'on' }, `kv ${kv} @4096`, async () => {
    const runs = [];
    for (let i = 0; i < 3; i++) runs.push(await chat(SHORT, 256));
    const tps = runs.reduce((s, r) => s + r.tps, 0) / runs.length;
    return { kv_gen_tps: Number(tps.toFixed(1)), label_text: `${tps.toFixed(1)} tok/s` };
  }));
}
for (const kv of ['f16', 'q8_0', 'q4_0']) {
  results.push(await measure({ ctx: 8192, ngl: 99, kv, fa: 'on' }, `kv ${kv} @8192`, async () => {
    const r = await chat(SHORT, 32);
    return { kv_gen_tps: Number(r.tps.toFixed(1)), label_text: `${r.tps.toFixed(1)} tok/s` };
  }));
}

// 5. Prefix cache reuse: same long prompt twice.
results.push(await measure({ ctx: 4096, ngl: 99, kv: 'q8_0', fa: 'on', cacheReuse: 256 }, 'prefix reuse', async () => {
  const first = await chat(LONG, 16);
  const second = await chat(LONG, 16);
  return { reuse_first_ms: Math.round(first.ms), reuse_second_ms: Math.round(second.ms), label_text: `${Math.round(first.ms)} ms then ${Math.round(second.ms)} ms` };
}));

fs.writeFileSync('bench/results.json', JSON.stringify({ model: MODEL, results: results.filter(Boolean) }, null, 2) + '\n');

const rows = results.filter(Boolean);
console.log('\n| config | measurement | VRAM |');
console.log('|---|---|---:|');
for (const r of rows) console.log(`| ${r.label} | ${r.label_text} | ${r.vram_bytes ? (r.vram_bytes / 1e9).toFixed(1) + ' GB' : '-'} |`);
