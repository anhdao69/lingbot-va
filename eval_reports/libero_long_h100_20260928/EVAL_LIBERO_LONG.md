Evaluation of robbyant/lingbot-va-posttrain-libero-long on LIBERO-LONG
================================================================

SSH access and execution within interactive Slurm job 4474 were verified on worker-3. The allocation has two NVIDIA H100 80GB GPUs and an unlimited Slurm time limit.

The isolated environment is .venv-libero-eval: Python 3.10.16, PyTorch 2.9.0+cu126, Diffusers 0.36.0, Transformers 4.55.2, NumPy 1.26.4, robosuite 1.4.0, MuJoCo 2.3.7. All 109 installed packages passed uv pip check. Exact installed versions are recorded in .eval_setup/requirements.lock.txt.

The checkpoint is in checkpoints/lingbot-va-posttrain-libero-long, pinned to Hugging Face revision 0e89d1e753019988aba484e8da2dc0810e264d9f. All seven safetensors files match published file sizes. Its index contains all 839 expected model tensors; the two extra patch_embedding tensors belong to the older convolutional embedding and are unused by the current MLP embedding. The transformer config was changed from attn_mode=flex to attn_mode=torch for inference; its original is saved in .eval_setup/transformer_config_original.json.

Official LIBERO-LONG demonstrations are in datasets/libero_10, pinned to dataset revision f13aa24a3da8c43c7225569f28c562979fa0e35a. All 10 HDF5 files were opened and verified to contain 50 demonstrations and seven action channels. Evaluation uses the simulator assets, BDDL task files and 50 fixed initial states per task from third_party/LIBERO. Every task passed an environment reset, initial-state load, five settling steps, and validation of both 128x128 camera images. The downloaded demonstrations are available separately and are not read during simulation evaluation.

The node lacks NVIDIA EGL graphics libraries. OSMesa is installed locally under .eval_setup/system-libs for headless rendering; no system packages were changed. Inference uses the H100 GPUs. .eval_setup/env.sh supplies all required paths and renderer settings. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 preserves loading of LIBERO's trusted, pinned legacy initial-state files with PyTorch 2.9.

Two small compatibility changes are recorded in .eval_setup/compatibility.patch: FlashAttention imports are optional when using PyTorch attention, and the client uses Python's json module instead of importing LeRobot's training dependencies solely for JSON writing. Inference wrappers set the local checkpoint path, server mode, loopback binding, and seed 0. The released action normalization, channel IDs, and diffusion settings are unchanged.

The separate smoke test completed successfully: task 0, initial state 0, 1/1 success, about 145 seconds, video saved. It is excluded from the full benchmark count.

The full run uses libero_10 (LIBERO-LONG), the original task order, 50 episodes per task, seed 0, five settling steps, and the repository client's 800-timestep loop limit. GPU 0 evaluates tasks 0–4; GPU 1 evaluates tasks 5–9. A total of 500 episodes is expected. This is one seeded run, not the paper's aggregate across multiple seeds.

The active output directory is outputs/libero_long_20260928T211222Z_seed0. Metrics and videos are saved after each episode. summary.json is refreshed every 30 seconds. run_status.txt distinguishes SMOKE_TEST, RUNNING_FULL_EVAL, COMPLETE, and FAILED. The orchestrator validates that all 500 episodes are present before marking COMPLETE and stops its model servers when finished. Logs are in logs/libero_long_20260928T211222Z_seed0, with orchestration and monitoring logs in logs/eval_orchestrator.log and logs/eval_monitor.log.

From the login node, check live progress with:

    srun --jobid=4474 --overlap --nodes=1 --ntasks=1 bash /mnt/data/vmo-ai-task/anhdh35/lingbot-va/.eval_setup/status.sh

From a shell already on worker-3:

    cd /mnt/data/vmo-ai-task/anhdh35/lingbot-va
    bash .eval_setup/status.sh

The runner is detached from this SSH connection and continues while allocation 4474 remains active. The 500-episode run is expected to take several hours. A new run can be launched using .eval_setup/run_eval.sh in a GPU allocation; it creates a separate output directory. Do not launch it while the current run is using ports 30056, 30057, 30061 and 30062.

Sources: https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long and https://github.com/Lifelong-Robot-Learning/LIBERO

Observed progress at 2026-09-28 21:19:05 UTC: the full evaluation was RUNNING_FULL_EVAL, with 2/500 episodes completed and 2 successes (one episode each from tasks 0 and 5). This is an early partial result, not the final benchmark success rate. Both H100s were actively executing inference, using approximately 25GB each. The local summary_snapshot.json is a copied snapshot; use the remote status command above for current results.
