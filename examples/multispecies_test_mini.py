#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:05:00 2020
@author: andrew
"""

from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager
inferelator_verbose_level(1)

DATA_DIR = '../data/haloarchaea_mini'
OUTPUT_DIR = '../haloarchaea_inference/'
#DATA_DIR = '/hpc/group/schmidlab/als185/inferelator_env/inferelator/data/haloarchaea'
#OUTPUT_DIR = '/hpc/group/schmidlab/als185/inferelator_env/haloarchaea_output_mini'

#PRIORS_FILE_NAME = 'gold_standard.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'prior.tsv'
#TF_LIST_FILE_NAME = 'tf_names.tsv'

HSAL_EXPRESSION = 'exp_1.tsv'
HSAL_METADATA = 'metadata1.tsv'
HSAL_PRIOR =  'prior.tsv'
HSAL_TFS = 'tf_list.tsv'

HVOL_EXPRESSION = 'exp_2.tsv'
HVOL_METADATA = 'metadata2.tsv'
HVOL_PRIOR = 'prior.tsv'
HVOL_TFS = 'tf_list.tsv'

CV_SEEDS = list(range(42, 43))

n_cores_local = 1
local_engine = True

if __name__ == '__main__' and local_engine:
    MPControl.set_multiprocess_engine("local")
    MPControl.client.processes = n_cores_local
    MPControl.connect()

cv_wrap = CrossValidationManager()

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Create a worker
worker = inferelator_workflow(regression="amusr", workflow="multitask")
worker.set_file_paths(input_dir=DATA_DIR, output_dir=OUTPUT_DIR, gold_standard_file = GOLD_STANDARD_FILE_NAME)

# Create tasks
task1 = worker.create_task(task_name="Hsal",
                           input_dir=DATA_DIR,
                           expression_matrix_file=HSAL_EXPRESSION,
                           tf_names_file=HSAL_TFS,
                           meta_data_file=HSAL_METADATA,
                           priors_file=HSAL_PRIOR,
                           workflow_type="tfa")
task1.set_file_properties(expression_matrix_columns_are_genes=False)

task2 = worker.create_task(task_name="Hvol",
                           input_dir=DATA_DIR,
                           expression_matrix_file=HVOL_EXPRESSION,
                           tf_names_file=HVOL_TFS,
                           meta_data_file=HVOL_METADATA,
                           priors_file=HVOL_PRIOR,
                           workflow_type="tfa")
task2.set_file_properties(expression_matrix_columns_are_genes=False)
worker.set_run_parameters(num_bootstraps=1)
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "halo_mini")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker


# Run
cv_wrap.run()