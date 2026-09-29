| Model | Self-attn ms | Block ms | Video ms | Action ms | Clean policy ms | Peak allocated GiB | Attn speedup | Policy speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 2129.5 | 5458.0 | 1882.3 | 4136.3 | 5866.1 | 23.805 | 1.000× | 1.000× |
| Local 50% | 1651.8 | 4768.9 | 1644.4 | 3647.8 | 5239.1 | 23.043 | 1.289× | 1.120× |
| Local 75% | 1494.1 | 4594.9 | 1609.8 | 3517.4 | 5082.1 | 22.636 | 1.425× | 1.154× |
| Local 100% | 813.0 | 4086.8 | 1324.3 | 3327.9 | 4374.9 | 22.289 | 2.619× | 1.341× |
| GDN 50% | 1889.9 | 5086.7 | 1798.7 | 3838.2 | 5678.2 | 23.095 | 1.127× | 1.033× |
| GDN 75% | 1717.1 | 4893.5 | 1743.6 | 3698.7 | 5365.0 | 22.713 | 1.240× | 1.093× |
| GDN 100% | 1053.4 | 4326.2 | 1472.4 | 3421.5 | 4507.3 | 22.383 | 2.021× | 1.301× |

Self-attention and block columns sum all 30 layers across all 72 denoising calls. Video/action spans include intervening scheduler/host dispatch between Transformer calls. These four columns use instrumented calls; clean policy times use separate calls without profiler hooks. Their totals need not equal clean policy latency. All times are milliseconds.

| Model | Persistent cache peak MiB | Cache saving | Observed cache-update ms | Policy calls/s | Action-equivalent Hz (with commit) | Full cycle ms | Full cycle speedup | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 1519.4 | 0.0% | 232.9 | 0.170 | 2.623 | 6099.0 | 1.000× | 23.865 |
| Local 50% | 759.7 | 50.0% | 224.5 | 0.191 | 2.928 | 5463.6 | 1.116× | 23.104 |
| Local 75% | 354.5 | 76.7% | 271.7 | 0.197 | 2.989 | 5353.8 | 1.139× | 22.697 |
| Local 100% | 0.0 | 100.0% | 217.7 | 0.229 | 3.484 | 4592.5 | 1.328× | 22.359 |
| GDN 50% | 849.7 | 44.1% | 245.8 | 0.176 | 2.701 | 5924.0 | 1.030× | 23.361 |
| GDN 75% | 492.5 | 67.6% | 238.7 | 0.186 | 2.855 | 5603.6 | 1.088× | 23.346 |
| GDN 100% | 180.0 | 88.2% | 242.9 | 0.222 | 3.368 | 4750.2 | 1.284× | 23.109 |

Persistent GDN cache peak includes committed **and** speculative states. Allocator peak also includes transient state-update/output tensors. Hz excludes simulator, transport, model load and reset. The policy emits 16 actions per steady-state call; action-equivalent Hz is not a sensor feedback rate.

| Model | Clean samples | Clean policy mean ± SD ms | Instrumented policy mean ms | Cross-attn ms | FFN ms | Mean one-block invocation ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 11 | 5866.1 ± 211.0 | 6032.8 | 645.6 | 277.6 | 2.527 |
| Local 50% | 11 | 5239.1 ± 372.0 | 5307.9 | 599.5 | 247.1 | 2.208 |
| Local 75% | 11 | 5082.1 ± 270.7 | 5140.6 | 584.0 | 243.4 | 2.127 |
| Local 100% | 11 | 4374.9 ± 81.6 | 4664.6 | 603.8 | 265.2 | 1.892 |
| GDN 50% | 11 | 5678.2 ± 171.6 | 5759.5 | 626.8 | 264.9 | 2.355 |
| GDN 75% | 11 | 5365.0 ± 239.0 | 5455.3 | 617.1 | 260.9 | 2.266 |
| GDN 100% | 11 | 4507.3 ± 34.6 | 4906.2 | 634.7 | 279.9 | 2.003 |

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
| smoke_dense_before | none | 0 | 0 | True | 158.9 |
| smoke_dense_before | none | 5 | 0 | True | 83.9 |
| smoke_gdn100_after | after | 0 | 0 | False | 363.8 |
| smoke_gdn100_after | after | 5 | 0 | False | 329.5 |
| smoke_gdn100_before | none | 0 | 0 | False | 338.9 |
| smoke_gdn100_before | none | 5 | 0 | False | 309.2 |
| smoke_gdn50_after | after | 0 | 0 | False | 392.0 |
| smoke_gdn50_after | after | 5 | 0 | False | 366.5 |
| smoke_gdn50_before | none | 0 | 0 | False | 399.5 |
| smoke_gdn50_before | none | 5 | 0 | False | 356.4 |
| smoke_gdn75_after | after | 0 | 0 | False | 388.7 |
| smoke_gdn75_after | after | 5 | 0 | False | 345.7 |
| smoke_gdn75_before | none | 0 | 0 | False | 384.8 |
| smoke_gdn75_before | none | 5 | 0 | False | 345.5 |
| smoke_local100_before | none | 0 | 0 | False | 344.5 |
| smoke_local100_matched | none | 5 | 0 | False | 300.3 |

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
