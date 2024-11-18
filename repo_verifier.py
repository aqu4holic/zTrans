import pandas as pd
from typing import List, Tuple, Dict, Any
import subprocess
from threading import Thread
import copy
import argparse

process_cnt: int = 10
thread_cnt: int = 20

threads: List[Any] = []
output_queue_str: List[Any] = [{} for i in range(thread_cnt)]

processes: List[Any] = []
output_queue: Dict[str, Any] = {i: None for i in range(process_cnt)}

data_prefix: str = 'data'
repo_prefix: str = f'{data_prefix}/repos'

parser = argparse.ArgumentParser(description = 'Process a file.')
parser.add_argument('--filename', nargs = '?', default = 'original.parquet', help = 'The name of the file to process')
parser.add_argument('--output', nargs = '?', default = 'sampled_code.parquet', help = 'The name of the file to output')
args = parser.parse_args()

data_name: str = args.filename
output_name: str = args.output

repo_df: pd.DataFrame = pd.read_parquet(f'{data_prefix}/{data_name}', engine = 'pyarrow')

# define template to crawl data
get_status_template: str = '''
git status {}/{}
'''

output: Dict[str, Any] = {}

def get_repo_status(repo_prefix: str, repo_name: str) -> str:
    get_repo_status_script: str = get_status_template.format(repo_prefix, repo_name, )
    sub: subprocess.CompletedProcess = subprocess.run(get_repo_status_script, shell = True, capture_output = True, encoding = 'utf-8', errors = 'ignore')

    status: str = sub.stdout.strip()
    err: str = sub.stderr.strip()

    return status, err

class get_repo_status_by_thread(Thread):
    def __init__(self, repo_prefix: str, repo_name: str):
        Thread.__init__(self)

        self.repo_prefix: str = repo_prefix
        self.repo_name: str = repo_name

    def run(self):
        repo_prefix: str = self.repo_prefix
        repo_name: str = self.repo_name

        status, err = get_repo_status(repo_prefix, repo_name)

        output[repo_name] = {
            'status': status,
            'err': err,
        }


def main():
    errors = []
    # retrieve unique repository names
    unique_repos: List[str] = repo_df['repoName'].unique().tolist()

    print(f'repo count: {len(unique_repos)}')
    print(f'-' * 50)
    print()

    for id in range(len(unique_repos)):
        repo_id: str = unique_repos[id]

        repo_owner, repo_name = repo_id.split('_', 1)

        thread_id: int = id % thread_cnt

        thread: get_repo_status_by_thread = get_repo_status_by_thread(repo_prefix, repo_id)
        threads.append(thread)

        if (len(threads) == thread_cnt) or (id == len(unique_repos) - 1):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # for key in output.keys():
            #     status, err = output[key]['status'], output[key]['err']

            #     if (len(err) > 0):
            #         print(f'Error: {err}')
            #         print(f'Repo: {key}')
            #         errors.append(key)
            #         print(f'-' * 50)

            threads.clear()
            # output.clear()

    output_df = pd.DataFrame.from_dict(output, orient = 'index')
    output_df.to_parquet(f'{data_prefix}/git_status.parquet', engine = 'pyarrow')

    with open(f'{data_prefix}/errors.txt', 'w') as f:
        for error in errors:
            f.write(f'{error}\n')

if (__name__ == '__main__'):
    main()

# time python repo_verifier.py --filename="original_filtered1.parquet" --output="errors.parquet"