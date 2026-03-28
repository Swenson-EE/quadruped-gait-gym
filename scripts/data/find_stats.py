import os
import glob
import pandas as pd
import numpy as np
import ast

from loggers.base.statistics_logger import Statistics
from shared.data.data_util import get_algorithm_directories, extract_number



def parse_np_array(cell):
    """Convert a string like '[-12.12 138.06 31.71]' into a numpy array"""
    if isinstance(cell, str):
        # remove brackets and split on spaces
        s = cell.strip("[]")
        numbers = [float(x) for x in s.split()]
        return np.array(numbers)
    elif isinstance(cell, (list, np.ndarray)):
        return np.array(cell)
    else:
        return np.array([cell])


def load_checkpoint_stats(file, sheets):
    dfs = {}
    
    excel = pd.ExcelFile(file)
        
    for sheet_name in excel.sheet_names:
        if sheet_name not in sheets:
            continue

        df = excel.parse(sheet_name)

        dfs[sheet_name] = {}
        

        for col in df.columns:
            if any(col.endswith(f"_{s.value}") for s in Statistics):
                name, stat = next(
                    (col.rsplit(f"_{s.value}", 1)[0], s)
                    for s in Statistics
                    if col.endswith(f"_{s.value}")
                )
                

                dfs[sheet_name][name] = {}
                dfs[sheet_name][name][stat] = {}

                arrays = df[col].apply(parse_np_array)
                stacked = np.stack(arrays.values)

                dfs[sheet_name][name] = {
                    'mean': np.mean(stacked, axis=0),
                    'std': np.std(stacked, axis=0),
                    'min': np.min(stacked, axis=0),
                    'max': np.max(stacked, axis=0)
                }
        
    return dfs


def build_dataframe(stats):
    # Get sheet names from the first checkpoint
    sheet_names = next(iter(stats.values())).keys()
    stats_keys = ["min", "max", "mean", "std"]

    sheet_dfs = {}

    for sheet in sheet_names:
        # Create a DataFrame directly from rows using a list comprehension
        df = pd.DataFrame(
            [
                [ckpt_data.get(sheet, {}).get(obs, {}).get(stat, None)
                 for obs in next(iter(stats.values()))[sheet].keys()
                 for stat in stats_keys]
                for ckpt_data in stats.values()
            ],
            index=stats.keys(),
            columns=pd.MultiIndex.from_product(
                [next(iter(stats.values()))[sheet].keys(), stats_keys],
                names=["Observation", "Stat"]
            )
        )
        df.index.name = "Checkpoint"
        sheet_dfs[sheet] = df

    return sheet_dfs
            
    



if __name__ == "__main__":

    for algorithm in get_algorithm_directories('saved'):
        data_dir = os.path.join('saved', algorithm, 'data')
        if not os.path.exists(data_dir):
            continue

        excel_files = [f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
        excel_files = sorted(excel_files, key=extract_number)

        stats = {}

        for file in excel_files:
            n = extract_number(file)
            checkpoint_stats = load_checkpoint_stats(file, ["Observations", "Components"])

            stats[n] = checkpoint_stats

        
        dfs = build_dataframe(stats)


        if not os.path.isdir('saved/overall'):
            os.makedirs('saved/overall', exist_ok=True)

        with pd.ExcelWriter(f"saved/overall/{algorithm}_overall.xlsx") as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name)



