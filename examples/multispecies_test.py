#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:05:00 2020
@author: andrew
"""

from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager
inferelator_verbose_level(1)

DATA_DIR = '../data/haloarchaea'
OUTPUT_DIR = '../haloarchaea_inference/'

#PRIORS_FILE_NAME = 'gold_standard.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'Hvol_prior_matrix.tsv'
#TF_LIST_FILE_NAME = 'tf_names.tsv'
HOMOLOGY_FILE_NAME = 'homology_file_hsal_hvol.tsv'

HSAL_EXPRESSION = '1154_normalized_GE_data_modern.tsv'
#HSAL_EXPRESSION = '1154_converted_hvol.tsv'
HSAL_METADATA = 'hsal_clean_sample_metadata.tsv'
HSAL_PRIOR = 'Hsal_prior_matrix.tsv'
#HSAL_PRIOR =  'Hsal_prior_matrix_converted_Hvol.tsv'
HSAL_TFS = 'combined_tfs.tsv'

HVOL_EXPRESSION = 'hvol_tpm_exp.tsv'
HVOL_METADATA = 'hvol_metadata.tsv'
HVOL_PRIOR = 'Hvol_prior_matrix.tsv'
HVOL_TFS = 'hvol_tfs.tsv'

CV_SEEDS = list(range(42, 43))

n_cores_local = 1
local_engine = True

if __name__ == '__main__' and local_engine:
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.set_processes(n_cores_local)
    MPControl.connect()


cv_wrap = CrossValidationManager()

# Assign variables for grid searchimmediate stack trace
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Create a worker
worker = inferelator_workflow(regression="multispecies", workflow="multispecies")
worker.set_file_paths(input_dir=DATA_DIR,
                      output_dir=OUTPUT_DIR,
                      homology_file = HOMOLOGY_FILE_NAME)
#gold_standard_file = GOLD_STANDARD_FILE_NAME,

# Create tasks
task1 = worker.create_task(task_name="Hsal",
                           input_dir=DATA_DIR,
                           expression_matrix_file=HSAL_EXPRESSION,
                           tf_names_file=HSAL_TFS,
                           meta_data_file=HSAL_METADATA,
                           gold_standard_file=HSAL_PRIOR,
                           priors_file = HSAL_PRIOR,
                           workflow_type="tfa")


task1.set_file_properties(expression_matrix_columns_are_genes=False)

task2 = worker.create_task(task_name="Hvol",
                           input_dir=DATA_DIR,
                           expression_matrix_file=HVOL_EXPRESSION,
                           tf_names_file=HVOL_TFS,
                           meta_data_file=HVOL_METADATA,
                           gold_standard_file=HVOL_PRIOR,
                           priors_file = HVOL_PRIOR,
                           workflow_type="tfa")
task2.set_file_properties(expression_matrix_columns_are_genes=False)
worker.set_run_parameters(num_bootstraps=1)
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "bsubtilis_1_2_MTL")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker


# Run
cv_wrap.run()



data = pd.read_csv("../data/haloarchaea/hvol_tpm_exp.tsv", sep="\t")
data = data.drop(["gene"], axis=1)