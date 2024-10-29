import pandas as pd
import numpy as np
import os
# from threading import Thread
from multiprocess import Process

seed = 18022004
np.random.seed(seed)

data_prefix = 'data'
repo_prefix = f'{data_prefix}/repos'

repo_df = pd.read_parquet(f'{data_prefix}/500_sampled_raw.parquet', engine = 'fastparquet')

unique_repos = repo_df['repoName'].unique()

git_string = 'git clone git@github.com:{}/{}.git {}/{}'

class clone_repo_by_thread(Process):
    def __init__(self, repo_id, repo_owner, repo_name, git_clone_string):
        Process.__init__(self)

        self.repo_id = repo_id
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.git_clone_string = git_clone_string

    def run(self):
        repo_id = self.repo_id
        repo_owner = self.repo_owner
        repo_name = self.repo_name
        git_clone_string = self.git_clone_string

        # print(git_clone_string)

        os.system(git_clone_string)

        print(f'Finished repo_id: {repo_id}')
        print('-' * 100)

thread_cnt = 15
threads = []

for id in range(len(unique_repos)):
    repo_id = unique_repos[id]

    repo_owner, repo_name = repo_id.split('_', 1)

    git_clone_string = git_string.format(repo_owner, repo_name, repo_prefix, repo_id)

    cloner = clone_repo_by_thread(repo_id, repo_owner, repo_name, git_clone_string)
    threads.append(cloner)

    if (len(threads) == thread_cnt) or (id == len(unique_repos) - 1):
        for i in threads:
            i.start()

        for i in threads:
            i.join()

        threads = []

print('Finished')