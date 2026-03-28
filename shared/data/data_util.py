

def get_algorithm_directories(dir: str):
    import os
    from shared.algorithm.algorithm_types import Algorithm

    return [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and any(algo in d for algo in Algorithm)]


def extract_number(filename: str):
    import os
    import re

    base = os.path.basename(filename)
    match = re.search(r'_(\d+)', base)  # number before any file extension
    return int(match.group(1)) if match else -1 
