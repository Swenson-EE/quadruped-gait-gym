from dataclasses import field
import os, shutil
import argparse

from algorithms.algorithm_types import Algorithm

def parse_algorithm(value: str) -> Algorithm:
    try:
        return Algorithm(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid algorithm '{value}'. Valid options {[algo.value for algo in Algorithm]}"
        )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithms", 
        type=parse_algorithm,
        nargs="+",
        required=False,
        choices=[algo.value for algo in Algorithm]
    )

    return parser.parse_args()


def clear_saved_data(algorithms: list[Algorithm],folder = 'saved'):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                if algorithms is None or any(algo.value in file_path for algo in algorithms):
                    shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    args = parse_args()
    clear_saved_data(args.algorithms)


