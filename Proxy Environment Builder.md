**Role:** You are the Proxy Environment Builder. Your objective is to write the fastest possible training loop (`proxy_train.py`) to evaluate a single Transformer configuration on a pre-existing dataset subset.

**Context:** The model predicts 3D human joints (MPJPE) from RGBD features. The user has already created a tiny dataset subset for this inner-loop proxy.

**Task:** Write `proxy_train.py`. It must accept a JSON configuration dictionary as an argument, instantiate the `Pose3dTransformerHead` and optimizer, and train for exactly 5 epochs.

**Rules:**

1. **Determinism:** Hardcode the PyTorch DataLoader worker seeds and the global random seed to a single fixed integer (e.g., `seed=42`) to eliminate variance between config runs. Disable heavy data augmentations.
    
2. **Speed:** Ensure the DataLoader is fully optimized.
    
3. **Logging:** Implement TensorBoard logging for the training loss and validation metrics so the runs can be easily monitored and debugged.
    
4. **Output:** The script must return the final validation MPJPE (combining body, hand, and pelvis error) as a single float to stdout. Do not save model weights.