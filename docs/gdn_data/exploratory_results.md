| Model | Self-attn ms | Block ms | Video ms | Action ms | Clean policy ms | Peak allocated GiB | Attn speedup | Policy speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 2100.7 | 5290.5 | 1803.9 | 4023.2 | 5683.4 | 23.805 | 1.000× | 1.000× |
| Local 50% | 1852.8 | 5195.6 | 1847.9 | 3917.5 | 5740.4 | 23.043 | 1.134× | 0.990× |
| Local 75% | 1568.1 | 4809.0 | 1718.8 | 3652.5 | 5366.4 | 22.636 | 1.340× | 1.059× |
| Local 100% | 842.0 | 4107.6 | 1381.8 | 3299.1 | 4575.9 | 22.289 | 2.495× | 1.242× |
| GDN 50% | 2029.5 | 5494.9 | 1917.0 | 4165.5 | 5894.5 | 23.095 | 1.035× | 0.964× |
| GDN 75% | 1677.3 | 4865.1 | 1717.1 | 3687.2 | 5360.5 | 22.713 | 1.252× | 1.060× |
| GDN 100% | 1077.4 | 4511.4 | 1481.6 | 3595.7 | 4938.0 | 22.383 | 1.950× | 1.151× |

Self-attention and block columns sum all 30 layers across all 72 denoising calls. Video/action spans include intervening scheduler/host dispatch between Transformer calls. These four columns use instrumented calls; clean policy times use separate calls without profiler hooks. Their totals need not equal clean policy latency. All times are milliseconds.

| Model | Persistent cache peak MiB | Cache saving | Observed cache-update ms | Policy calls/s | Action-equivalent Hz (with commit) | Full cycle ms | Full cycle speedup | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 1519.4 | 0.0% | 234.4 | 0.176 | 2.704 | 5917.8 | 1.000× | 23.865 |
| Local 50% | 759.7 | 50.0% | 241.7 | 0.174 | 2.675 | 5982.2 | 0.989× | 23.104 |
| Local 75% | 354.5 | 76.7% | 272.5 | 0.186 | 2.837 | 5638.9 | 1.049× | 22.697 |
| Local 100% | 0.0 | 100.0% | 227.0 | 0.219 | 3.331 | 4802.9 | 1.232× | 22.359 |
| GDN 50% | 849.7 | 44.1% | 249.1 | 0.170 | 2.604 | 6143.6 | 0.963× | 23.361 |
| GDN 75% | 492.5 | 67.6% | 235.5 | 0.187 | 2.859 | 5596.0 | 1.058× | 23.346 |
| GDN 100% | 180.0 | 88.2% | 247.0 | 0.203 | 3.086 | 5185.0 | 1.141× | 23.109 |

Persistent GDN cache peak includes committed **and** speculative states. Allocator peak also includes transient state-update/output tensors. Hz excludes simulator, transport, model load and reset. The policy emits 16 actions per steady-state call; action-equivalent Hz is not a sensor feedback rate.

| Model | Clean samples | Clean policy mean ± SD ms | Instrumented policy mean ms | Cross-attn ms | FFN ms | Mean one-block invocation ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 11 | 5683.4 ± 247.4 | 5841.1 | 683.7 | 254.4 | 2.449 |
| Local 50% | 11 | 5740.4 ± 298.7 | 5780.1 | 664.9 | 293.3 | 2.405 |
| Local 75% | 11 | 5366.4 ± 303.2 | 5384.8 | 627.5 | 272.9 | 2.226 |
| Local 100% | 11 | 4575.9 ± 220.8 | 4693.6 | 628.9 | 284.1 | 1.902 |
| GDN 50% | 11 | 5894.5 ± 290.2 | 6097.7 | 663.7 | 294.2 | 2.544 |
| GDN 75% | 11 | 5360.5 ± 232.1 | 5416.3 | 598.6 | 249.2 | 2.252 |
| GDN 100% | 11 | 4938.0 ± 210.5 | 5092.2 | 645.6 | 285.2 | 2.089 |

### Synthetic eager attention (ms)

| Query tokens | Historical tokens | Dense SDPA | Dense gather + SDPA | Local | Local + GDN read | Local + GDN read/update |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1000 | 0.0339 | 0.1464 | 0.0257 | 0.0570 | 0.1881 |
| 16 | 2000 | 0.0472 | 0.1996 | 0.0248 | 0.0571 | 0.1797 |
| 16 | 5000 | 0.0852 | 0.4217 | 0.0233 | 0.0512 | 0.1674 |
| 16 | 10000 | 0.1480 | 0.7813 | 0.0229 | 0.0509 | 0.1630 |
| 16 | 20000 | 0.2758 | 1.4987 | 0.0244 | 0.0535 | 0.1797 |
| 128 | 1000 | 0.0347 | 0.1322 | 0.0236 | 0.0564 | 0.2727 |
| 128 | 2000 | 0.0480 | 0.2069 | 0.0232 | 0.0557 | 0.2626 |
| 128 | 5000 | 0.0873 | 0.4325 | 0.0246 | 0.0571 | 0.2700 |
| 128 | 10000 | 0.1503 | 0.7901 | 0.0230 | 0.0549 | 0.2616 |
| 128 | 20000 | 0.2884 | 1.5143 | 0.0258 | 0.0596 | 0.2724 |

These CUDA-event intervals include eager Python launch gaps, as deployment does. They are not CUDA-graph-only hardware throughput. History construction is excluded from repeated reads; commit measurements include gate materialization and FLA recurrence.

| Query tokens | Update/read kernel | ms |
|---:|---|---:|
| 16 | chunk | 0.7659 |
| 16 | recurrent | 0.1476 |
| 16 | read_torch | 0.0803 |
| 16 | read_triton | 0.0295 |
| 128 | chunk | 0.5655 |
| 128 | recurrent | 0.2120 |
| 128 | read_torch | 0.0774 |
| 128 | read_triton | 0.0329 |

### Actual dense LIBERO history

| Chunk | Video history before | Action history before | Profiled policy ms | Self-attn ms | Cross-attn ms | FFN ms | Block ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 128 | 8312.4 | 2714.5 | 1558.4 | 357.1 | 7257.0 |
| 3 | 432 | 560 | 6395.9 | 2399.1 | 672.6 | 295.4 | 5788.1 |
| 8 | 1152 | 1280 | 6331.0 | 2467.5 | 657.0 | 278.8 | 5755.4 |
| 15 | 2160 | 2160 | 6661.7 | 2431.7 | 721.5 | 331.4 | 6034.9 |

Chunk 0 is cold and excluded from steady-state conclusions. This first actual rollout used an earlier, heavier profiler (including nested SDPA events); its absolute times are not the clean comparative timings above.

### LIBERO smoke checks

| Model | Adaptation | Task | Initial state | Success | Episode seconds |
|---|---|---:|---:|---|---:|
| dense_libero_profile | none | 0 | 0 | True | 169.2 |
| smoke_gdn50_before | none | 0 | 0 | False | 399.5 |
| smoke_gdn50_before | none | 5 | 0 | False | 356.4 |
| smoke_gdn75_before | none | 0 | 0 | False | 384.8 |
| smoke_local100_before | none | 0 | 0 | False | 344.5 |
| smoke_local100_before | none | 5 | 0 | False | 313.3 |

### Teacher-forced attention alignment

| Split | Local NMSE | GDN before NMSE | GDN after NMSE |
|---|---:|---:|---:|
| train chunks 1/5/10 | 0.82823 | 0.82510 | 0.79191 |
| held-out chunk 14 | 0.86169 | 0.85895 | 0.83497 |

| Held-out phase | Local NMSE | GDN before NMSE | GDN after NMSE |
|---|---:|---:|---:|
| video | 0.17761 | 0.18598 | 0.19678 |
| action | 1.54577 | 1.53192 | 1.47316 |

Alignment: 36 Adam steps/layer, learning rate 0.025, 2160 trainable parameters, 48.7 s including setup/evaluation. No original checkpoint weights were loaded into the optimizer process.
