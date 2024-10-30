import pandas as pd
from typing import List, Tuple, Dict, Any
import subprocess
from threading import Thread
import copy

process_cnt: int = 10
thread_cnt: int = 20

threads: List[Any] = [[] for i in range(process_cnt)]
output_queue_str: List[Any] = [{} for i in range(process_cnt)]

processes: List[Any] = []
output_queue: Dict[str, Any] = {i: None for i in range(process_cnt)}

data_prefix: str = 'data'
repo_prefix: str = f'{data_prefix}/repos'

data_name: str = '500_sampled_no_code.parquet'

repo_df: pd.DataFrame = pd.read_parquet(f'{data_prefix}/{data_name}', engine = 'fastparquet')

# define template to crawl data
get_prev_commit_template: str = '''
cd ./{}/{}

git rev-parse {}^
'''

get_diff_2_commit_template: str = '''
cd ./{}/{}

git diff --name-only {} {}
'''

get_file_at_commit_template: str = '''
cd ./{}/{}

git show {}:{}
'''

get_diff_file_template: str = '''
cd ./{}/{}

git diff {}..{} -- {}
'''

necessary_cols: List[str] = [
    'id',
    'fromLib',
    'toLib',
    'repoName',
    'repoOwner',
    'repoSplitName',
    'startCommit',
    'endCommit',
    'fileName',
    'startCode',
    'endCode',
    'diff',
    'startCommitChanges',
    'endCommitChanges',
]
sample_template: Dict[str, Any] = {k: None for k in necessary_cols}
final_df: pd.DataFrame = pd.DataFrame(columns = necessary_cols)
# final_df.set_index(id)

def get_prev_commit(repo_prefix: str, repo_name: str, changed_commit: str) -> str:
    get_prev_commit_script: str = get_prev_commit_template.format(repo_prefix, repo_name, changed_commit)
    sub: subprocess.CompletedProcess = subprocess.run(get_prev_commit_script, shell = True, capture_output = True, encoding = 'utf-8', errors = 'ignore')

    prev_commit: str = sub.stdout.strip()

    return prev_commit

def get_diff_2_commit(repo_prefix: str, repo_name: str, commit1: str, commit2: str) -> List[str]:
    get_diff_2_commit_script: str = get_diff_2_commit_template.format(repo_prefix, repo_name, commit1, commit2)
    sub: subprocess.CompletedProcess = subprocess.run(get_diff_2_commit_script, shell = True, capture_output = True, encoding = 'utf-8', errors = 'ignore')

    diff: str = sub.stdout
    diff_files: List[str] = diff.split('\n')

    return diff_files

def get_start_end_commit_code(repo_prefix: str, repo_name: str, file_name: str, start_commit: str, end_commit: str) -> Tuple[str, str]:
    try:
        get_file_script: str = get_file_at_commit_template.format(repo_prefix, repo_name, start_commit, file_name)
        sub: subprocess.CompletedProcess = subprocess.run(get_file_script, shell = True, capture_output = True, encoding = 'utf-8', errors = 'ignore')
        start_commit_code: str = sub.stdout
    except Exception as e:
        start_commit_code: str = ''


    try:
        get_file_script: str = get_file_at_commit_template.format(repo_prefix, repo_name, end_commit, file_name)
        sub: subprocess.CompletedProcess = subprocess.run(get_file_script, shell = True, capture_output = True, encoding='utf-8', errors='ignore')
        end_commit_code: str = sub.stdout
    except Exception as e:
        end_commit_code: str = ''

    return start_commit_code, end_commit_code

def get_diff_file(repo_prefix: str, repo_name: str, file_name: str, start_commit: str, end_commit: str) -> str:
    get_diff_file_script: str = get_diff_file_template.format(repo_prefix, repo_name, start_commit, end_commit, file_name)
    sub: subprocess.CompletedProcess = subprocess.run(get_diff_file_script, shell = True, capture_output = True, encoding = 'utf-8', errors = 'ignore')
    diff: str = sub.stdout

    return diff

class get_file_by_thread(Thread):
    def __init__(self, _pid: int, id: int, repo_prefix: str, repo_name: str, file_name: str, start_commit: str, end_commit: str):
        Thread.__init__(self)

        self._pid = _pid
        self.id = id
        self.repo_prefix = repo_prefix
        self.repo_name = repo_name
        self.file_name = file_name
        self.start_commit = start_commit
        self.end_commit = end_commit

    def run(self):
        _pid = self._pid
        id = self.id
        repo_prefix = self.repo_prefix
        repo_name = self.repo_name
        file_name = self.file_name
        start_commit = self.start_commit
        end_commit = self.end_commit

        start_code, end_code = get_start_end_commit_code(repo_prefix = repo_prefix, repo_name = repo_name, file_name = file_name,
                                                                start_commit = start_commit, end_commit = end_commit)

        diff = get_diff_file(repo_prefix = repo_prefix, repo_name = repo_name, file_name = file_name,
                            start_commit = start_commit, end_commit = end_commit)

        output_queue_str[_pid][file_name] = [id, start_code, end_code, diff]

def str_normalize(x: str) -> str:
    if (x is None):
        return ''
    elif (len(x) == 0):
        return ''

    return x

def create_data_rows(samples: pd.DataFrame, repo_name: str, sample_template: Dict[str, Any] = sample_template, sample_cnt: int = 0) -> Tuple[int, pd.DataFrame]:
    sample_template.update({
        # 'id': sample_cnt,
        'repoName': repo_name,
        'fromLib': samples.iloc[0]['fromLib'],
        'toLib': samples.iloc[0]['toLib'],
        'repoOwner': samples.iloc[0]['repoOwner'],
        'repoSplitName': samples.iloc[0]['repoSplitName'],
        'startCommit': samples.iloc[0]['startCommit'],
        'endCommit': samples.iloc[0]['endCommit'],
        'startCode': '',
        'endCode': '',
        'startCommitChanges': samples.iloc[0]['startCommitChanges'],
        'endCommitChanges': samples.iloc[0]['endCommitChanges']
    })

    # get unique startCommit values for this repository's samples
    changed_commits: List[str] = samples['startCommit'].unique().tolist()

    res_df: pd.DataFrame = pd.DataFrame(columns = necessary_cols)

    # get the diff of each commit and its previous commib
    for commit_id in range(len(changed_commits)):
        changed_commit: str = changed_commits[commit_id]

        # get the previous commit hash and the diff
        prev_commit: str = get_prev_commit(repo_prefix = repo_prefix, repo_name = repo_name, changed_commit = changed_commit)
        diff_files: str = get_diff_2_commit(repo_prefix = repo_prefix, repo_name = repo_name,
                                        commit1 = changed_commit, commit2 = prev_commit)

        for file_name in diff_files:
            try:
                start_code, end_code = get_start_end_commit_code(repo_prefix = repo_prefix, repo_name = repo_name, file_name = file_name,
                                                                start_commit = prev_commit, end_commit = changed_commit)
                diff = get_diff_file(repo_prefix = repo_prefix, repo_name = repo_name, file_name = file_name,
                                    start_commit = prev_commit, end_commit = changed_commit)
            except Exception as e:
                print(e)
                print(f'file: {file_name}')
                print(f'start: {prev_commit}, end: {changed_commit}')
                print(f'start code: {start_code}')
                print(f'end code: {end_code}')
                print('-' * 50)

                return None

            sample_template['id'] = sample_cnt
            sample_template['fileName'] = file_name
            sample_template['startCode'], sample_template['endCode'] = str_normalize(start_code), str_normalize(end_code)
            sample_template['diff'] = diff

            res_df = pd.concat([res_df, pd.DataFrame([sample_template], columns = necessary_cols)], ignore_index = True)

            sample_cnt += 1

    return sample_cnt, res_df

def create_data_rows_by_thread(_pid: int, samples: pd.DataFrame, repo_name: str, sample_template: Dict[str, Any] = sample_template, sample_cnt: int = 0) -> Tuple[int, pd.DataFrame]:
    current_sample_template = copy.deepcopy(sample_template)
    current_sample_template.update({
        # 'id': sample_cnt,
        'repoName': repo_name,
        'fromLib': samples.iloc[0]['fromLib'],
        'toLib': samples.iloc[0]['toLib'],
        'repoOwner': samples.iloc[0]['repoOwner'],
        'repoSplitName': samples.iloc[0]['repoSplitName'],
        'startCommit': samples.iloc[0]['startCommit'],
        'endCommit': samples.iloc[0]['endCommit'],
        'startCode': '',
        'endCode': '',
        'startCommitChanges': samples.iloc[0]['startCommitChanges'],
        'endCommitChanges': samples.iloc[0]['endCommitChanges']
    })

    # get unique startCommit values for this repository's samples
    changed_commits: List[str] = samples['startCommit'].unique().tolist()

    res_df: pd.DataFrame = pd.DataFrame(columns = necessary_cols)

    threads[_pid] = []
    output_queue_str[_pid] = {}
    file_queue = []

    # get the diff of each commit and its previous commib
    for commit_id in range(len(changed_commits)):
        changed_commit: str = changed_commits[commit_id]

        # get the previous commit hash and the diff
        prev_commit: str = get_prev_commit(repo_prefix = repo_prefix, repo_name = repo_name, changed_commit = changed_commit)
        diff_files: str = get_diff_2_commit(repo_prefix = repo_prefix, repo_name = repo_name,
                                        commit1 = changed_commit, commit2 = prev_commit)

        for file_name in diff_files:
            thread: Thread = get_file_by_thread(_pid = _pid, id = sample_cnt, repo_prefix = repo_prefix, repo_name = repo_name, file_name = file_name,
                                                start_commit = prev_commit, end_commit = changed_commit)
            threads[_pid].append(thread)
            file_queue.append(file_name)

            sample_cnt += 1

            if ((len(threads[_pid]) == thread_cnt) or (file_name == diff_files[-1])):
                for thread in threads[_pid]:
                    thread.start()

                for thread in threads[_pid]:
                    thread.join()

                for key in file_queue:
                    current_sample_template['id'] = output_queue_str[_pid][key][0]
                    current_sample_template['fileName'] = key
                    current_sample_template['startCode'], current_sample_template['endCode'] = str_normalize(output_queue_str[_pid][key][1]), str_normalize(output_queue_str[_pid][key][2])
                    current_sample_template['diff'] = str_normalize(output_queue_str[_pid][key][3])

                    res_df = pd.concat([res_df, pd.DataFrame([current_sample_template], columns = necessary_cols)], ignore_index = True)

                output_queue_str[_pid] = {}
                threads[_pid] = []
                file_queue = []

    return sample_cnt, res_df

class create_data_rows_by_process(Thread):
    def __init__(self, _pid: int, repo_name: str, samples: pd.DataFrame, sample_cnt: int = 0):
        Thread.__init__(self)

        self._pid = _pid
        self.repo_name = repo_name
        self.samples = samples
        self.sample_cnt = sample_cnt

    def run(self):
        _pid = self._pid
        repo_name = self.repo_name
        samples = self.samples
        sample_cnt = self.sample_cnt

        sample_cnt, res_df = create_data_rows_by_thread(_pid = _pid, samples = samples, repo_name = repo_name, sample_cnt = 0)

        output_queue[_pid] = res_df

        print(f'finished: {repo_name}')
        print('-' * 50)

# retrieve unique repository names
unique_repos: List[str] = repo_df['repoName'].unique().tolist()

print(f'repo count: {len(unique_repos)}')
print(f'-' * 50)
print()

for repo_name_id in range(len(unique_repos)):
    repo_name: str = unique_repos[repo_name_id]

    # filter the DataFrame for the current repository's samples
    samples: pd.DataFrame = repo_df[repo_df['repoName'] == repo_name]
    res_df: pd.DataFrame = None

    _pid: int = repo_name_id % process_cnt
    proc: Thread = create_data_rows_by_process(_pid = _pid, repo_name = repo_name, samples = samples)
    processes.append(proc)

    # sample_cnt, res_df = create_data_rows(samples = samples, repo_name = repo_name, sample_cnt = 0)
    # print(len(res_df))
    # break

    if ((len(processes) == process_cnt) or (repo_name_id == len(unique_repos) - 1)):
        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        for key in range(process_cnt):
            res_df = output_queue[key]
            final_df = pd.concat([final_df, res_df], ignore_index = True)

        output_queue = {i: None for i in range(process_cnt)}
        processes = []

        final_df['id'] = final_df.index
        final_df.to_parquet(f'{data_prefix}/first_dataset.parquet')

        print()
        print('()' * 25)
        print(' ' * 20 + f'finished: {repo_name_id + 1}/{len(unique_repos)}')
        print(' ' * 20 + f'len: {len(final_df)}')
        print(' ' * 20 + f'checkpointed!')
        print('()' * 25)
        print()

final_df['id'] = final_df.index
print(f'finished: {len(final_df)} line(s)')
final_df.to_parquet(f'{data_prefix}/first_dataset.parquet')
