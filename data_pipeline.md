# init
classified_res.csv -> original.parquet (added repoSplitName, repoOwner)

# filter errors
origin.parquet -> filter error repos (non existent repos) -> original_filtered1.parquet
original_filtered1.parquet -> repo_verifier.py -> original_filtered2.parquet
original_filtered2.parquet -> official_original.parquet

# create dataset
official_original.parquet -> phat_sample_data.ipynb -> migration_{others, log, test}.parquet

we only take others
migration_others.parquet -> data_builder.py -> migration_others_code.parquet

filter class code
migration_others_code.parquet -> data_preview.ipynb (get_code_df_endswith) -> migration_others_class_code.parquet