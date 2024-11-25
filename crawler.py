import pandas as pd
import numpy as np
import os
from threading import Thread
import subprocess

seed = 18022004
np.random.seed(seed)

data_prefix = 'data'
repo_prefix = f'{data_prefix}/repos'

data_name = 'original_filtered1.parquet'

repo_df = pd.read_parquet(f'{data_prefix}/{data_name}', engine = 'pyarrow')

# with open(f'{data_prefix}/repos.txt', 'r') as f:
#     unique_repos = f.readlines()

unique_repos = repo_df['repoName'].unique()

git_string = 'git clone git@github.com:{}/{}.git {}/{}'

errors = []

class clone_repo_by_thread(Thread):
    def __init__(self, repo_id, repo_owner, repo_name, git_clone_string):
        Thread.__init__(self)

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
        sub: subprocess.CompletedProcess = subprocess.Popen(git_clone_string, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE, encoding = 'utf-8', errors = 'ignore')

        stdout, stderr = sub.communicate()

        if (sub.returncode != 0):
            print(f'Error in repo_id: {repo_id}')
            print(stderr)
            errors.append(repo_id)

        print(f'Finished repo_id: {repo_id}')
        print('-' * 100)

thread_cnt = 20
threads = []

for id in range(len(unique_repos)):
    repo_id = unique_repos[id]

    repo_owner, repo_name = repo_id.split('_', 1)

    git_clone_string = git_string.format(repo_owner, repo_name, repo_prefix, repo_id)

    cloner = clone_repo_by_thread(repo_id, repo_owner, repo_name, git_clone_string)
    cloner.start()
    threads.append(cloner)

    if (len(threads) == thread_cnt) or (id == len(unique_repos) - 1):
        print('Starting threads from {} to {}'.format(id - len(threads) + 1, id))

        # for i in threads:
        #     i.start()

        for i in threads:
            i.join()

        threads = []

with open(f'{data_prefix}/errors_git_clone.txt', 'w') as f:
    for error in errors:
        f.write(f'{error}\n')

print('Finished')