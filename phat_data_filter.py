import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import difflib
import pyarrow.parquet as pq

seed = 18022004
np.random.seed(seed)

data_prefix: str = 'data'
repo_prefix: str = f'{data_prefix}/repos'

# data_name = 'full_others_code_dataset.parquet'
# data_name = 'data_method_level.parquet'
data_name = 'data_method.parquet'

data_df: pd.DataFrame = pd.read_parquet(f'{data_prefix}/{data_name}', engine = 'pyarrow')
# table = pq.read_table(f'{data_prefix}/{data_name}')
# data_df = table.to_pandas()

# print(len(data_df))

data_df = data_df.reset_index(drop = True)
data_df['id'] = data_df.index

def get_diff(string1, string2):
    diff = difflib.unified_diff(
        string1.splitlines(),
        string2.splitlines(),
        lineterm = ''
    )
    return '\n'.join(diff)

data_df['method_diff'] = None

for id in range(len(data_df)):
    line = data_df.iloc[id]

    method_b, method_a = line['methods_before'], line['methods_after']
    diff = get_diff(method_b, method_a)

    data_df.at[id, 'method_diff'] = diff

data_df.to_parquet(f'{data_prefix}/data_method_diff.parquet', engine = 'pyarrow')
