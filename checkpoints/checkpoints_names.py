
import os

from algorithms.algorithm_types import Algorithm



def encode_metadata(**kwargs):
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = "x".join(map(str, v))  # [64,64,32] -> "64x64x32"
        parts.append(f"{k}-{v}")
    return "_".join(parts)



def next_checkpoint(algo: Algorithm, save_dir='saved', **metadata):
    """
    Returns the path to the next available checkpoint n
    """

    save_dir = f"{save_dir}/{algo}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    meta_str = encode_metadata(**metadata)
    prefix = f"{algo}_{meta_str}" if meta_str else f"{algo}"

    existing = [f for f in os.listdir(save_dir) if f.startswith(prefix)]

    numbers = []
    for f in existing:
        try:
            n = int(f.split("_")[-1].split(".zip")[0])
            numbers.append(n)
        except:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    name = f"{prefix}_{next_num}"

    return os.path.join(save_dir, name), name



def get_latest_checkpoint(algo: Algorithm, save_dir="saved", **metadata):
    """
    Returns the path to the latest checkpoint (highest _N) in the directory.
    If none exist, returns None.
    """

    save_dir = f"{save_dir}/{algo}/checkpoints"
    if not os.path.exists(save_dir):
        return None

    meta_str = encode_metadata(**metadata)
    prefix = f"{algo}_{meta_str}" if meta_str else f"{algo}"

    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith(prefix) and f.endswith(".zip")
    ]

    numbers = []
    for f in existing:
        try:
            n = int(f.split("_")[-1].split(".zip")[0])
            numbers.append((n, f))
        except:
            continue

    if not numbers:
        return None

    latest_file = max(numbers, key=lambda x: x[0])[1]
    return os.path.join(save_dir, latest_file)
