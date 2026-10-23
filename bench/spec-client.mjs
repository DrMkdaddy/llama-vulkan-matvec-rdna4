// Client for spec-decode measurements: hits llama-server /completion and reports
// decode throughput plus draft acceptance (llama.cpp returns draft_n/draft_n_accepted
// in timings when speculative decoding is enabled).
import { performance } from 'node:perf_hooks';

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, ...v] = a.replace(/^--/, '').split('=');
    return [k, v.join('=') || 'true'];
  })
);

const PORT = args.port || '8090';
const RUNS = Number(args.runs || 3);
const MAX_TOKENS = Number(args['max-tokens'] || 256);
const CONCURRENT = Number(args.concurrent || 1);
const PROMPT = args.prompt === 'json'
  ? 'Write a JSON array of 20 objects describing planets. Each object has keys name, diameter_km, and moons. Output only the JSON, no prose.'
  : 'Explain in detail how a modern CPU pipeline works, covering fetch, decode, out-of-order execution, register renaming, and branch prediction.';

const url = `http://127.0.0.1:${PORT}/completion`;
const rows = [];

for (let i = 0; i < RUNS; i++) {
  const started = performance.now();
  const bodies = Array.from({ length: CONCURRENT }, (_, k) =>
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: PROMPT,
        n_predict: MAX_TOKENS,
        temperature: 0.7,
        cache_prompt: false,
        stream: false,
        seed: 1000 + k,
      }),
    }).then((r) => r.json())
  );
  const jsons = await Promise.all(bodies);
  const wall = performance.now() - started;

  const timings = jsons.map((j) => j.timings).filter(Boolean);
  if (!timings.length) {
    console.log('no timings in response:', JSON.stringify(jsons[0]).slice(0, 300));
    continue;
  }
  const sum = (f) => timings.reduce((s, t) => s + f(t), 0);
  const predicted = sum((t) => t.predicted_n || 0);
  const row = {
    run: i + 1,
    concurrent: CONCURRENT,
    prompt_tps: +(sum((t) => t.prompt_per_second || 0) / timings.length).toFixed(1),
    decode_tps: +(sum((t) => t.predicted_per_second || 0)).toFixed(1),
    aggregate_tps: +(predicted / (wall / 1000)).toFixed(1),
    predicted_n: predicted,
    draft_n: sum((t) => t.draft_n ?? 0),
    draft_accepted: sum((t) => t.draft_n_accepted ?? 0),
    accept_rate: sum((t) => t.draft_n ?? 0)
      ? +((sum((t) => t.draft_n_accepted ?? 0) / sum((t) => t.draft_n ?? 0)) * 100).toFixed(1)
      : null,
    wall_ms: Math.round(wall),
  };
  rows.push(row);
  console.log(JSON.stringify(row));
}

const decodeRuns = rows.filter((r) => r.decode_tps);
if (decodeRuns.length) {
  const avg = (f) => decodeRuns.reduce((s, r) => s + f(r), 0) / decodeRuns.length;
  const totalDraft = decodeRuns.reduce((s, r) => s + r.draft_n, 0);
  const totalAcc = decodeRuns.reduce((s, r) => s + r.draft_accepted, 0);
  console.log(
    JSON.stringify({
      summary: true,
      runs: decodeRuns.length,
      concurrent: CONCURRENT,
      decode_tps_avg: +avg((r) => r.decode_tps).toFixed(1),
      decode_tps_max: Math.max(...decodeRuns.map((r) => r.decode_tps)),
      aggregate_tps_avg: +avg((r) => r.aggregate_tps).toFixed(1),
      accept_rate_pct: totalDraft ? +((totalAcc / totalDraft) * 100).toFixed(1) : null,
      draft_n_total: totalDraft,
    })
  );
}
