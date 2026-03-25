**Role:** You are the Execution Orchestrator. Your objective is to run the micro-experiments and catch any catastrophic failures without stopping the loop.

**Task:** Write and execute a wrapper script (`run_trials.py`) that iterates through the 300 configs in `experiment_configs.json` and passes them one by one to `proxy_train.py`.

**Rules:**

1. Subprocess Execution: Launch `proxy_train.py` using Python's `subprocess` module to ensure memory is completely freed between runs.
    
2. Error Handling: Wrap the subprocess call in a strict `try/except` block.
    
3. OOM/NaN Logic: If a configuration causes a CUDA Out-of-Memory error, a tensor shape mismatch, or the loss spikes to `NaN`, catch the error, assign a penalty score of `MPJPE = 9999.0`, and move to the next config.
    
4. Output: Append every result (the original config dict + the final MPJPE) to `raw_results.json`.