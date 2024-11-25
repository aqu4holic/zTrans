#!/usr/bin/env bash

time python data_builder.py --filename="2000_sample_no_code.parquet" --output="2000_sample_dataset.parquet"
time python data_builder.py --filename="phat_500_log_sample_no_code.parquet" --output="500_log_sample_dataset.parquet"
time python data_builder.py --filename="phat_500_test_sample_no_code.parquet" --output="500_test_sample_dataset.parquet"
time python data_builder.py --filename="phat_500_others_sample_no_code.parquet" --output="500_others_sample_dataset.parquet"

time python data_builder.py --filename="phat_full_others_no_code.parquet" --output="full_others_code_dataset.parquet"
time python data_builder.py --filename="migration_others.parquet" --output="migration_others_code.parquet"