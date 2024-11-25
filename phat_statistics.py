# %% [markdown]
# # init

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import difflib
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import time
import argparse

seed = 18022004
np.random.seed(seed)

# %%
data_prefix: str = 'data'
repo_prefix: str = f'{data_prefix}/repos'

model_id: str = 'deepseek-ai/deepseek-coder-6.7b-instruct'

# %% [markdown]
# # statistics

# %% [markdown]
# ## init data

# %%

parser = argparse.ArgumentParser(description = 'Process a file.')
parser.add_argument('--filename', nargs = '?', default = 'sampled_no_code.parquet', help = 'The name of the file to process')
parser.add_argument('--output', nargs = '?', default = 'sampled_code.parquet', help = 'The name of the file to output')
args = parser.parse_args()

# data_name = 'ori_data_method_treesitter.parquet'
data_name: str = args.filename
output_name: str = args.output

start_time: float = time.time()
data_df: pd.DataFrame = pd.read_parquet(f'{data_prefix}/{data_name}', engine = 'pyarrow')
end_time: float = time.time()
print(f'Data loaded: {data_df.shape}')
print(f'Time: {end_time - start_time:.2f}s')
print('-' * 100)

# %% [markdown]
# # statistics

# %% [markdown]
# * special cases
#
# group by migration:
# - number of classes
# -  number of methods
#
# group by class:
# - number of methods (add, remove, total)
# - number of lines (add, remove, total)
#
# group by method:
# - number of lines (add, remove, total)
# - number of tokens (add, remove, total)

# %% [markdown]
# ## by migration

# %% [markdown]
# ## by class

# %% [markdown]
############################################################################# by method

# %%
data_df['added'] = None
data_df['removed'] = None

data_df['method_before_token'] = None
data_df['method_after_token'] = None
data_df['method_before_line'] = None
data_df['method_after_line'] = None

data_df['added_token'] = None
data_df['removed_token'] = None
data_df['added_line'] = None
data_df['removed_line'] = None

data_df['changed_token'] = None
data_df['changed_line'] = None

# %%
def build_tokenizer(model_id: str = model_id) -> AutoTokenizer:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True,)

    return tokenizer

tokenizer: AutoTokenizer = build_tokenizer(model_id = model_id)
print('Tokenizer built')
print('-' * 100)

# %%
from threading import Thread

def get_diff(string1: str, string2: str) -> str:
    # Normalize by removing leading/trailing whitespace and replacing tabs with spaces
    normalized1: List[str] = [line.strip().replace('\t', '') for line in string1.splitlines()]
    normalized2: List[str] = [line.strip().replace('\t', '') for line in string2.splitlines()]

    # Generate the diff
    diff: str = difflib.unified_diff(
        normalized1,
        normalized2,
        lineterm = ''
    )
    return '\n'.join(diff)

def extract_diff_changes(diff_str: str) -> Tuple[List[str], List[str], str, str]:
    added_lines: List[str] = []
    removed_lines: List[str] = []

    # Split the diff into lines
    lines: List[str] = diff_str.splitlines()

    # Iterate through the lines
    for line in lines:
        if line.startswith('+') and not line.startswith('+++'):
            # Line added (exclude the '+++' line indicating the file name)
            added_lines.append(line[1:].strip())
        elif line.startswith('-') and not line.startswith('---'):
            # Line removed (exclude the '---' line indicating the file name)
            removed_lines.append(line[1:].strip())

    # Join the lines with '\n'
    added_str: str = '\n'.join(added_lines)
    removed_str: str = '\n'.join(removed_lines)

    return added_lines, removed_lines, added_str, removed_str

def get_token_count(tokenizer: AutoTokenizer, text: str) -> List[int]:
    return len(tokenizer.encode(text, return_tensors = 'pt').to('cpu')[0])

output: Dict = {}

class get_method_data_by_thread(Thread):
    def __init__(self, sample: pd.DataFrame, id: int, tokenizer: AutoTokenizer) -> None:
        Thread.__init__(self)

        self.sample = sample
        self.id = id
        self.tokenizer = tokenizer

    def run(self) -> None:
        id: int = self.sample['id']
        sample: pd.DataFrame = self.sample
        tokenizer: AutoTokenizer = self.tokenizer

        method_before: str = sample['method_before']
        method_after: str = sample['method_after']

        method_before_line: int = len(method_before.split('\n'))
        method_after_line: int = len(method_after.split('\n'))

        method_before_token: int = get_token_count(tokenizer = tokenizer, text = method_before)
        method_after_token: int = get_token_count(tokenizer = tokenizer, text = method_after)

        diff: str = sample['method_diff']
        added_lines, removed_lines, added_str, removed_str = extract_diff_changes(diff)
        added_token: int = get_token_count(tokenizer = tokenizer, text = added_str)
        removed_token: int = get_token_count(tokenizer = tokenizer, text = removed_str)

        changed_token: int = added_token + removed_token
        changed_line: int = max(len(added_lines), len(removed_lines))

        output[id] = {
            'added': added_str,
            'removed': removed_str,

            'method_before_token': method_before_token,
            'method_after_token': method_after_token,
            'method_before_line': method_before_line,
            'method_after_line': method_after_line,

            'added_token': added_token,
            'removed_token': removed_token,
            'added_line': len(added_lines),
            'removed_line': len(removed_lines),

            'changed_token': changed_token,
            'changed_line': changed_line,
        }

# %%
def method_statistics_single_thread(data_df: pd.DataFrame) -> pd.DataFrame:
    for id in tqdm(range(len(data_df)), desc = 'Processing method', total = len(data_df)):
        sample = data_df.iloc[id]

        method_before: str = sample['method_before']
        method_after: str = sample['method_after']

        method_before_line: int = len(method_before.split('\n'))
        method_after_line: int = len(method_after.split('\n'))

        method_before_token: int = get_token_count(tokenizer = tokenizer, text = method_before)
        method_after_token: int = get_token_count(tokenizer = tokenizer, text = method_after)

        diff: str = sample['method_diff']
        added_lines, removed_lines, added_str, removed_str = extract_diff_changes(diff)
        added_token: int = get_token_count(tokenizer = tokenizer, text = added_str)
        removed_token: int = get_token_count(tokenizer = tokenizer, text = removed_str)

        changed_token: int = added_token + removed_token
        changed_line: int = max(len(added_lines), len(removed_lines))

        # print(method_before_token_count, method_after_token_count, added_str_token_count, removed_str_token_count)

        # add to the df
        data_df.at[id, 'added'] = added_str
        data_df.at[id, 'removed'] = removed_str

        data_df.at[id, 'method_before_token'] = method_before_token
        data_df.at[id, 'method_after_token'] = method_after_token
        data_df.at[id, 'method_before_line'] = len(method_before_line)
        data_df.at[id, 'method_after_line'] = len(method_after_line)

        data_df.at[id, 'added_token'] = added_token
        data_df.at[id, 'removed_token'] = removed_token
        data_df.at[id, 'added_line'] = len(added_lines)
        data_df.at[id, 'removed_line'] = len(removed_lines)

        data_df.at[id, 'changed_token'] = changed_token
        data_df.at[id, 'changed_line'] = changed_line

    return data_df

def method_statistics_multi_thread(data_df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    threads = []
    thread_cnt = 50

    for id in tqdm(range(len(data_df)), desc = 'Processing method', total = len(data_df)):
        sample = data_df.iloc[id]
        thread = get_method_data_by_thread(sample = sample, id = id, tokenizer = tokenizer)
        thread.start()
        threads.append(thread)

        if (len(threads) % thread_cnt == 0 or id == len(data_df) - 1):
            # for thread in threads:
            #     thread.start()
            for thread in threads:
                thread.join()

            for sample_id in output.keys():
                data_df.at[sample_id, 'added'] = output[sample_id]['added']
                data_df.at[sample_id, 'removed'] = output[sample_id]['removed']

                data_df.at[sample_id, 'method_before_token'] = output[sample_id]['method_before_token']
                data_df.at[sample_id, 'method_after_token'] = output[sample_id]['method_after_token']
                data_df.at[sample_id, 'method_before_line'] = output[sample_id]['method_before_line']
                data_df.at[sample_id, 'method_after_line'] = output[sample_id]['method_after_line']

                data_df.at[sample_id, 'added_token'] = output[sample_id]['added_token']
                data_df.at[sample_id, 'removed_token'] = output[sample_id]['removed_token']
                data_df.at[sample_id, 'added_line'] = output[sample_id]['added_line']
                data_df.at[sample_id, 'removed_line'] = output[sample_id]['removed_line']

                data_df.at[sample_id, 'changed_token'] = output[sample_id]['changed_token']
                data_df.at[sample_id, 'changed_line'] = output[sample_id]['changed_line']

            output.clear()
            threads.clear()

        # if ((id + 1) % 10000 == 0):
        #     data_df.to_parquet(f'{data_prefix}/phat_filtered_method.parquet', engine = 'pyarrow')
        #     print(f'Saved {id + 1} samples @ {data_prefix}/phat_filtered_method.parquet')
        #     print('-' * 100)

    start_time = time.time()
    data_df.to_parquet(f'{data_prefix}/{output_name}', engine = 'pyarrow')
    end_time = time.time()
    print(f'Saved {len(data_df)} samples @ {data_prefix}/{output_name}')
    print(f'Time: {end_time - start_time:.2f}s')
    print('-' * 100)

    return data_df

method_statistics_multi_thread(data_df = data_df, tokenizer = tokenizer)

# last_df.to_parquet(f'{data_prefix}/phat_filtered_method.parquet', engine = 'pyarrow')