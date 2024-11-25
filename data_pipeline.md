# init
classified_res.csv -> original.parquet (added repoSplitName, repoOwner)

# filter errors
origin.parquet -> filter error repos (non existent repos) -> original_filtered1.parquet
original_filtered1.parquet -> repo_verifier.py -> original_filtered2.parquet
original_filtered2.parquet -> official_original.parquet

# create dataset
official_original.parquet -> phat_sample_data.ipynb -> migration_{others, log, test}.parquet
13382

we only take others
migration_others.parquet -> data_builder.py -> migration_others_code.parquet
7981

filter class code
migration_others_code.parquet -> data_preview.ipynb (get_code_df_endswith) -> migration_others_class_code.parquet
151739

filter changes in import only + blank codes
migration_others_class_code.parquet -> quang_statistics.ipynb (get_import_df) -> migration_others_class_code_no_import.parquet
80673

split to methods
migration_others_class_code_no_import.parquet -> phat_parser.py -> migration_others_method.parquet
33961

split no special and no special
migration_others_method_no_code.parquet -> quang_statistic -> migration_other_method_no_code_filtered_special.parquet
154552

migration_others_method_no_code.parquet -> quang_statistic -> migration_other_method_no_code_special.parquet