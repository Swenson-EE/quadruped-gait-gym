import os
import glob
import pandas as pd
import numpy as np

from collections import defaultdict

from loggers.base.statistics_logger import Statistics
from shared.data.data_util import get_algorithm_directories, extract_number


def parse_np_array(cell):
    """Robust parsing for Excel-stored arrays"""

    if isinstance(cell, str):
        s = cell.strip()

        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()

            if not s:
                return np.array([])

            # Normalize separators (handle commas + spaces)
            s = s.replace(",", " ")

            tokens = s.split()

            values = []
            for t in tokens:
                try:
                    values.append(float(t))
                except ValueError:
                    # Skip bad tokens instead of crashing
                    continue

            return np.array(values, dtype=float)

        # fallback: scalar string
        try:
            return np.array([float(s)])
        except:
            return np.array([])

    elif isinstance(cell, (int, float, np.number)):
        return np.array([cell], dtype=float)

    elif isinstance(cell, (list, np.ndarray)):
        return np.array(cell, dtype=float)

    return np.array([])


# -----------------------------
# Shape handling (robust)
# -----------------------------
def pad_and_stack(arrays):
    """Pad arrays to same shape and stack"""

    arrays = [np.atleast_1d(a) for a in arrays]

    max_len = max(a.shape[0] for a in arrays)

    padded = [
        np.pad(a, (0, max_len - a.shape[0]), mode="constant")
        for a in arrays
    ]

    return np.stack(padded)


# -----------------------------
# Core Logic
# -----------------------------
def load_checkpoint_stats(file, sheets):
    dfs = {}

    excel = pd.ExcelFile(file)

    for sheet_name in excel.sheet_names:
        if sheet_name not in sheets:
            continue

        df = excel.parse(sheet_name)
        dfs[sheet_name] = {}

        for col in df.columns:

            # Skip non-stat columns
            stat_match = next(
                (s for s in Statistics if col.endswith(f"_{s.value}")),
                None
            )

            if stat_match is None:
                continue

            name = col.rsplit(f"_{stat_match.value}", 1)[0]

            arrays = df[col].apply(parse_np_array)

            try:
                stacked = pad_and_stack(arrays.values)
            except Exception as e:
                print(f"[ERROR] {file} | {col}: {e}")
                continue

            dfs[sheet_name][name] = {
                "mean": np.mean(stacked, axis=0),
                "std": np.std(stacked, axis=0),
                "min": np.min(stacked, axis=0),
                "max": np.max(stacked, axis=0),
            }

    return dfs


# -----------------------------
# DataFrame Builder
# -----------------------------
def build_dataframe(stats):
    sheet_names = next(iter(stats.values())).keys()
    stats_keys = ["min", "max", "mean", "std"]

    sheet_dfs = {}

    for sheet in sheet_names:
        observations = set()

        for ckpt_data in stats.values():
            observations.update(ckpt_data.get(sheet, {}).keys())

        observations = sorted(observations)

        rows = []

        for ckpt, ckpt_data in stats.items():
            row = []

            for obs in observations:
                for stat in stats_keys:
                    val = ckpt_data.get(sheet, {}).get(obs, {}).get(stat, None)

                    if isinstance(val, np.ndarray):
                        val = val.tolist()

                    row.append(val)

            rows.append(row)

        df = pd.DataFrame(
            rows,
            index=stats.keys(),
            columns=pd.MultiIndex.from_product(
                [observations, stats_keys],
                names=["Observation", "Stat"]
            )
        )

        df.index.name = "Checkpoint"
        sheet_dfs[sheet] = df

    return sheet_dfs


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    for algorithm in get_algorithm_directories("saved"):
        data_dir = os.path.join("saved", algorithm, "data")

        if not os.path.exists(data_dir):
            continue

        excel_files = [
            f for f in glob.glob(os.path.join(data_dir, "*.xlsx"))
            if not os.path.basename(f).startswith("~$")
        ]

        excel_files = sorted(excel_files, key=extract_number)

        stats = {}

        for file in excel_files:
            checkpoint = extract_number(file)

            checkpoint_stats = load_checkpoint_stats(
                file,
                ["Observations", "Components"]  # skip Rewards
            )

            stats[checkpoint] = checkpoint_stats

        dfs = build_dataframe(stats)

        os.makedirs("saved/overall", exist_ok=True)

        output_path = f"saved/overall/{algorithm}_overall.xlsx"

        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name)

        print(f"[INFO] Saved: {output_path}")