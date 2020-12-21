#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:16:52 2020

@author: andrew
"""

from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager
from inferelator import crossvalidation_workflow
from inferelator import utils, workflow, crossvalidation_workflow

inferelator_verbose_level(1)

#DATA_DIR = '../data/haloarchaea'
#OUTPUT_DIR = '~/haloarchaea_inference/'
DATA_DIR = '../inferelator_env/inferelator_augmented/data/haloarchaea'
OUTPUT_DIR = '../inferelator/inferelator_augmented/haloarchaea_inference/'


#PRIORS_FILE_NAME = 'gold_standard.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'Hsal_Hvol_combined_augmented_prior_matrix.tsv'
#TF_LIST_FILE_NAME = 'tf_names.tsv'

HSAL_EXPRESSION = '1154_converted_hvol_augmented.tsv'
HSAL_METADATA = 'hsal_clean_sample_metadata.tsv'
HSAL_PRIOR =  'Hsal_Hvol_combined_augmented_prior_matrix.tsv'
HSAL_TFS = 'tf_list_augmented.tsv'

HVOL_EXPRESSION = 'hvol_tpm_exp_augmented.tsv'
HVOL_METADATA = 'hvol_metadata.tsv'
HVOL_PRIOR = 'Hsal_Hvol_combined_augmented_prior_matrix.tsv'
HVOL_TFS = 'tf_list_augmented.tsv'

def set_up_workflow_final(wkf):
    wkf.set_file_paths(input_dir = DATA_DIR,
                       output_dir = OUTPUT_DIR,
                       priors_file = HSAL_PRIOR,
                       tf_names_file = HSAL_TFS,
                       gold_standard_file = HSAL_TFS)
    task1 = worker.create_task(task_name = "Hsal",
                               workflow_type = "tfa",
                               expression_matrix_file = HSAL_EXPRESSION,
                               meta_data_file = HSAL_METADATA)
    task2 = worker.create_task(task_name = "Hvol",
                               workflow_type = "tfa",
                               expression_matrix_file = HVOL_EXPRESSION,
                               meta_data_file = HVOL_METADATA)
    task1.set_file_properties(expression_matrix_columns_are_genes=False)
    task2.set_file_properties(expression_matrix_columns_are_genes=False)

def set_up_cv_seeds(wkf):
    cv = crossvalidation_workflow.CrossValidationManager(wkf)
    cv.add_gridsearch_parameter('random_seed', list(range(42,43)))
    return cv


def set_up_dask(n_jobs=1):
    MPControl.set_multiprocess_engine("dask-cluster")
    MPControl.client.use_default_configuration("duke_dcc_schmidlab", n_jobs=n_jobs)
    MPControl.client.add_worker_conda("")
    MPControl.client.add_slurm_command_line("--acount==schmidlab")
    MPControl.client.set_job_size_params(walltime="48:00:00")
    


if __name__ == '__main__':
    set_up_dask()
    utils.Debug.vprint("Testing preprocessing", level=0)
    
    worker = inferelator_workflow(regression = "amusr", workflow="multitask")
    set_up_workflow_final(worker)
    worker.set_run_parameters(num_bootstraps=5)
    
    worker.run()


"""
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

task2 = worker.create_task(task_name="Hsal2",
                           input_dir=DATA_DIR,
                           expression_matrix_file=HVOL_EXPRESSION,
                           tf_names_file=HVOL_TFS,
                           meta_data_file=HVOL_METADATA,
                           priors_file=HVOL_PRIOR,
                           workflow_type="tfa")
task2.set_file_properties(expression_matrix_columns_are_genes=False)
worker.set_run_parameters(num_bootstraps=1)
#worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "Double_Hsal")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker


# Run
cv_wrap.run()
"""