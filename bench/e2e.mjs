// End-to-end latency and throughput against a running llama-server, using the
// server's own timings (prompt_ms, predicted_per_second) rather than wall clock.
//
//   node bench/e2e.mjs
//
// Env: E2E_URL, E2E_MODEL, E2E_N (runs), E2E_MAX (output tokens).
import { performance } from 'node:perf_hooks';

const URL = process.env.E2E_URL || 'http://127.0.0.1:8082/v1/chat/completions';
const MODEL = process.env.E2E_MODEL || 'deepseek-r1-7b';
const N = Number(process.env.E2E_N || 3);
const MAX = Number(process.env.E2E_MAX || 256);
const PROMPT = 'The following is a system log excerpt. ' +
  'Request accepted, worker dispatched, response streamed back to the client over the wire. '.repeat(120) +
  '\n\nIn three sentences, explain what a hash table is.';

for (let i = 0; i < N; i++) {
  const t0 = performance.now();
  const res = await fetch(URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: PROMPT }],
      max_tokens: MAX,
      temperature: 0,
    }),
  });
  const j = await res.json();
  const total = performance.now() - t0;
  const t = j.timings || {};
  console.log(JSON.stringify({
    run: i + 1,
    prompt_tokens: j.usage?.prompt_tokens,
    prompt_ms: Math.round(t.prompt_ms || 0),
    prefill_tps: Math.round(t.prompt_per_second || 0),
    gen_tokens: j.usage?.completion_tokens,
    decode_tps: Number((t.predicted_per_second || 0).toFixed(1)),
    total_ms: Math.round(total),
  }));
}
