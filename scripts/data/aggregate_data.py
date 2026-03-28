import os
import glob
import pandas as pd

from shared.data.data_util import get_algorithm_directories, extract_number



if __name__ == "__main__":
    algorithm_directories = get_algorithm_directories('saved')

    for algorithm in algorithm_directories:
        data_dir = os.path.join('saved', algorithm, 'data')
        if not os.path.exists(data_dir):
            continue
        
        excel_files = [f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
        excel_files = sorted(excel_files, key=extract_number)
        
        dfs = [pd.read_excel(f) for f in excel_files]
        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df.to_excel(f"saved/aggregate/{algorithm}_aggregate.xlsx", index=False)

