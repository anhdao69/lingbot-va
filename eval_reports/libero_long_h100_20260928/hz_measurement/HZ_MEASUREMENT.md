Hz measured for LingBot-VA LIBERO-LONG on H100
=============================================

Measured on 2026-09-28 using the unchanged evaluation configuration: bfloat16, PyTorch attention, one H100 per model, 20 video denoising steps and 50 action denoising steps. Each steady inference request returns 16 seven-dimensional action vectors.

| Measurement | Result |
|---|---:|
| Generation request rate | 0.204 requests/sec |
| Mean generation latency for 16 actions | 4.898 sec |
| Median / p95 generation latency | 5.008 / 5.200 sec |
| Action-equivalent generation throughput | 3.267 actions/sec |
| Mean observation/KV-cache update | 0.211 sec |
| Action-equivalent throughput including KV update | 3.132 actions/sec |
| Active evaluation control loop, GPU 0 (task 0) | 2.252 actions/sec |
| Active evaluation control loop, GPU 1 (task 5) | 2.523 actions/sec |

Generation and cache-update latency used time.perf_counter with torch.cuda.synchronize immediately before and after each operation. Five warmup chunks were excluded; 20 measured chunks (320 action vectors) used recorded real LIBERO RGB observations and the same VA_Server inference/cache-update code. This measures computation through action postprocessing, including both video prediction and action prediction. It excludes simulator stepping, websocket transport, model loading and episode reset/text encoding. GPU 0's existing evaluation model and client were temporarily suspended with their state retained, so they did not compete for GPU 0 compute. GPU 1 continued its evaluation. Recorded observations were replayed to measure speed; this extra benchmark did not add evaluation trials or success-rate results. Progress logging was suppressed during timing.

The active evaluation rates came from 83 full cycles on GPU 0 and 82 full cycles on GPU 1, before the measurement pause. Each interval spans generation, 16 actual simulation action steps, observation encoding/KV update and websocket roundtrips. The first two cycles of each episode, terminal partial chunks, resets and video writing were excluded. This is active control-loop throughput, not whole-job throughput including initialization and episode transitions. The renderer is CPU OSMesa.

The model generates action chunks: a rate of 3.27 actions/sec is obtained as 16 / 4.90, and is not 3.27 separate policy queries per second. The generation query rate is 0.204 Hz. With the full simulator loop, new chunks arrive at 0.141–0.158 Hz in the observed tasks.

GPU 0 evaluation was suspended from 2026-09-28T21:28:52Z to 2026-09-28T21:31:21Z (149.6 sec). It resumed with the same server/client processes. Both evaluation servers were confirmed to advance after resumption, with no inference tracebacks. The 500-episode evaluation continues.

Raw results: model_hz.json, live_control_hz.json and samples.jsonl. The original remote files and benchmark log are in outputs/libero_long_20260928T211222Z_seed0/hz_measurement. Benchmark code is in .eval_setup/hz_benchmark.py; isolation/resumption is controlled by .eval_setup/hz_run.py with a separate recovery watchdog.
