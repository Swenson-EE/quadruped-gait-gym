import os
import glob
import pandas as pd
from collections import defaultdict

from shared.data.data_util import get_algorithm_directories, extract_number



if __name__ == "__main__":
    aggregate_path = "saved/aggregate"
    algorithm_directories = get_algorithm_directories('saved')
    

    for algorithm in algorithm_directories:
        data_dir = os.path.join('saved', algorithm, 'data')
        if not os.path.exists(data_dir):
            continue
        
        excel_files = [f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
        excel_files = sorted(excel_files, key=extract_number)

        sheet_groups = defaultdict(list)

        for f in excel_files:
            sheets = pd.read_excel(f, sheet_name = None)

            for sheet_name, df in sheets.items():
                sheet_groups[sheet_name].append(df)
                        

        if not os.path.exists('saved/aggregate'):
            os.makedirs("saved/aggregate", exist_ok=True)

        output_path = f"{aggregate_path}/{algorithm}_aggregate.xlsx"

        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df_list in sheet_groups.items():
                combined_df = pd.concat(df_list, ignore_index=True)
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

