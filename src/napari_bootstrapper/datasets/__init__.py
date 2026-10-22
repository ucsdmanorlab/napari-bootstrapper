import numpy as np


def stack_samples(samples):
    """Batch samples as numpy arrays instead of tensors.

    On macOS, tensors sent back from DataLoader workers use torch's
    file_system sharing, which starts one torch_shm_manager per worker.
    Each manager keeps running after its worker exits and makes the
    worker's shutdown wait its full 5 s timeout.
    """
    return [np.stack(arrays) for arrays in zip(*samples)]
