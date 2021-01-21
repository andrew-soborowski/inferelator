"""
Run Multitask Network Inference with TFA-AMuSR.
"""
import copy
import gc
import warnings
import numpy as np
from anndata import AnnData

from inferelator.utils import Debug
from inferelator import workflow
from inferelator import amusr_workflow
from inferelator.regression import amusr_regression
from inferelator.postprocessing.results_processor_mtl import ResultsProcessorMultiTask
from inferelator.postprocessing.results_processor_ms import ResultsProcessorMultiSpecies
from inferelator.amusr_workflow import create_task_data_class
from inferelator.amusr_workflow import create_task_data_object
from inferelator.utils import InferelatorDataLoader


TRANSFER_ATTRIBUTES = ['count_minimum', 'preprocessing_workflow', 'input_dir']
NON_TASK_ATTRIBUTES = ["gold_standard_file", "random_seed", "num_bootstraps"]


class MultispeciesLearningWorkflow(amusr_workflow.MultitaskLearningWorkflow):
    """
    Class that implements multitask learning. Handles loading and validation of multiple data packages.
    """
    Multispecies_identifier = True
    _regulator_expression_filter = "intersection"
    _target_expression_filter = "union"

    
    # Task-specific data
    _n_tasks = None
    _task_design = None
    _task_response = None
    _task_bootstraps = None
    _task_priors = None
    _task_names = None
    _task_objects = None

    task_results = None

    # Axis labels to keep
    _targets = None
    _regulators = None

    # Multi-task result processor
    _result_processor_driver = ResultsProcessorMultiSpecies

    @property
    def _num_obs(self):
        if self._task_objects is not None:
            return sum([t if t is not None else 0 for t in map(lambda x: x._num_obs, self._task_objects)])
        else:
            return None

    @property
    def _num_genes(self):
        if self._task_objects is not None:
            return max([t if t is not None else 0 for t in map(lambda x: x._num_genes, self._task_objects)])
        else:
            return None

    @property
    def _num_tfs(self):
        if self._task_objects is not None:
            return max([t if t is not None else 0 for t in map(lambda x: x._num_tfs, self._task_objects)])
        else:
            return None

    def set_task_filters(self, regulator_expression_filter=None, target_expression_filter=None):
        """
        Set the filtering criteria for regulators and targets between tasks

        :param regulator_expression_filter:
            "union" includes regulators which are present in any task,
            "intersection" includes regulators which are present in all tasks
        :type regulator_expression_filter: str, optional
        :param target_expression_filter:
            "union" includes targets which are present in any task,
            "intersection" includes targets which are present in all tasks
        :type target_expression_filter: str, optional
        """

        self._set_without_warning("_regulator_expression_filter", regulator_expression_filter)
        self._set_without_warning("_target_expression_filter", target_expression_filter)

    def startup_run(self):
        """
        Load data.

        This is called when `.startup()` is run. It is not necessary to call separately.
        """

        self.get_data()
        self.validate_data()

    def get_data(self):
        # Task data has expression & metadata and may have task-specific files for anything else
        self._load_tasks()

        # Priors, gold standard, tf_names, and gene metadata will be loaded if set
        self.read_homology()
        self.read_tfs()
        self.read_priors()
        self.read_genes()
       # self.create_tf_homology()
    def startup_finish(self):
        """
        Process task data and priors.

        This is called when `.startup()` is run. It is not necessary to call separately.
        """

        # Make sure tasks are set correctly
        #self._process_default_priors() Shouldn't ever have default priors or gs
        self._process_task_priors()
        self._process_task_data()


    def create_task(self, task_name=None, input_dir=None, expression_matrix_file=None, meta_data_file=None,
                    tf_names_file=None, priors_file=None, gene_names_file=None, gene_metadata_file=None, 
                    gold_standard_file=None, workflow_type="single-cell", **kwargs):
        """
        Create a task object and set any arguments to this function as attributes of that task object. TaskData objects
        are stored internally in _task_objects.

        :param task_name: A descriptive name for this task
        :type task_name: str
        :param input_dir: A path containing the input files
        :type input_dir: str
        :param expression_matrix_file: Path to the expression data
        :type expression_matrix_file: str
        :param meta_data_file: Path to the meta data
        :type meta_data_file: str, optional
        :param tf_names_file: Path to a list of regulator names to include in the model
        :type tf_names_file: str
        :param priors_file: Path to a prior data file
        :type priors_file: str
        :param gene_metadata_file: Path to a genes annotation file
        :type gene_metadata_file: str, optional
        :param gene_names_file: Path to a list of genes to include in the model (optional)
        :type gene_names_file: str, optional
        :param gold_standard_file: Path to a gold standard file
        :type gold_standard_file: str, optional
        :param workflow_type: The type of workflow for data preprocessing.
            "tfa" uses the TFA workflow,
            "single-cell" uses the Single-Cell TFA workflow
        :type workflow_type: str, `inferelator.BaseWorkflow` subclass
        :param kwargs: Any additional arguments are assigned to the task object.
        :return: Returns a task reference which can be additionally modified by calling any valid Workflow function to
            set task parameters
        :rtype: TaskData instance
        """

        # Create a TaskData object from a workflow and set the formal arguments into it
        task_object = create_task_data_object(workflow_class=workflow_type)
        task_object.task_name = task_name
        task_object.input_dir = input_dir if input_dir is not None else self.input_dir
        task_object.expression_matrix_file = expression_matrix_file
        task_object.meta_data_file = meta_data_file
        task_object.tf_names_file = tf_names_file
        task_object.priors_file = priors_file
        task_object.gene_names_file = gene_names_file
        task_object.gene_metadata_file = gene_metadata_file
        task_object.gold_standard_file = gold_standard_file
        
        # Warn if there is an attempt to set something that isn't supported
        msg = "Task-specific {} is not supported. This setting will be ignored. Set this in the parent workflow."
        for bad in NON_TASK_ATTRIBUTES:
            if bad in kwargs:
                del kwargs[bad]
                warnings.warn(msg.format(bad))

        # Pass forward any kwargs (raising errors if they're for attributes that don't exist)
        for attr, val in kwargs.items():
            if hasattr(task_object, attr):
                setattr(task_object, attr, val)
                task_object.str_attrs.append(attr)
            else:
                raise ValueError("Argument {attr} cannot be set as an attribute".format(attr=attr))

        if self._task_objects is None:
            self._task_objects = [task_object]
        else:
            self._task_objects.append(task_object)

        Debug.vprint(str(task_object), level=0)

        return task_object

    def validate_data(self):
        """
        Make sure that the data that's loaded is acceptable

        This is called when `.startup()` is run. It is not necessary to call separately.

        :raises ValueError: Raises a ValueError if any tasks have invalid priors or gold standard structures
        """
        no_gold_standard = sum(map(lambda x: x.gold_standard is None, self._task_objects))
        if no_gold_standard > 0:
            raise ValueError("{n} tasks have no gold standards".format(n=no_gold_standard))
        #if self.gold_standard is None:
         #   raise ValueError("A gold standard must be provided to `gold_standard_file` in MultiTaskLearningWorkflow")

        # Check to see if there are any tasks which don't have priors
        no_priors = sum(map(lambda x: x.priors_data is None, self._task_objects))
        if no_priors > 0 and self.priors_data is None:
            raise ValueError("{n} tasks have no priors (no default prior is set)".format(n=no_priors))

    def _process_default_priors(self):
        """
        Process the default priors in the parent workflow for crossvalidation or shuffling
        """

        priors = self.priors_data if self.priors_data is not None else self.gold_standard.copy()

        # Crossvalidation
        if self.split_gold_standard_for_crossvalidation:
            priors, self.gold_standard = self.prior_manager.cross_validate_gold_standard(priors, self.gold_standard,
                                                                                         self.cv_split_axis,
                                                                                         self.cv_split_ratio,
                                                                                         self.random_seed)
        # Filter to regulators
        if self.tf_names is not None:
            priors = self.prior_manager.filter_to_tf_names_list(priors, self.tf_names)

        # Filter to targets
        if self.gene_names is not None:
            priors = self.prior_manager.filter_priors_to_genes(priors, self.gene_names)

        # Shuffle labels
        if self.shuffle_prior_axis is not None:
            priors = self.prior_manager.shuffle_priors(priors, self.shuffle_prior_axis, self.random_seed)

        # Reset the priors_data in the parent workflow if it exists
        self.priors_data = priors if self.priors_data is not None else None

    def _process_task_priors(self):
        """
        Process & align the default priors for crossvalidation or shuffling
        """
        for task_obj in self._task_objects:

            # Set priors if task-specific priors are not present
            if task_obj.priors_data is None and self.priors_data is None:
                raise ValueError("No priors exist in the main workflow or in tasks")
            elif task_obj.priors_data is None:
                task_obj.priors_data = self.priors_data.copy() 

            # Set gene names if task-specific gene names is not present
            if task_obj.gene_names is None:
                task_obj.gene_names = copy.copy(self.gene_names)

            # Set tf_names if task-specific tf names are not present
            if task_obj.tf_names is None:
                task_obj.tf_names = copy.copy(self.tf_names)

            # Process priors in the task data
            '''
            task_obj.process_priors_and_gold_standard(gold_standard=task_obj.gold_standard,
                                                      cv_flag=self.split_gold_standard_for_crossvalidation,
                                                      cv_axis=self.cv_split_axis,
                                                      shuffle_priors=self.shuffle_prior_axis)
'''
    def _process_task_data(self):
        """
        Preprocess the individual task data using the TaskData worker into task design and response data. Set
        self.task_design, self.task_response, self.task_bootstraps with lists which contain
        DataFrames.

        Also set self.regulators and self.targets with pd.Indexes that correspond to the genes and tfs to model
        This is chosen based on the filtering strategy set in self.target_expression_filter and
        self.regulator_expression_filter
        """
        self._task_design, self._task_response = [], []
        self._task_bootstraps, self._task_names, self._task_priors = [], [], []
        targets, regulators = [], []

        # Iterate through a list of TaskData objects holding data
        for task_id, task_obj in enumerate(self._task_objects):
            # Get task name from Task
            task_name = task_obj.task_name if task_obj.task_name is not None else str(task_id)

            task_str = "Processing task #{tid} [{t}] {sh}"
            Debug.vprint(task_str.format(tid=task_id, t=task_name, sh=task_obj.data.shape), level=1)

            # Run the preprocessing workflow
            task_obj.startup_finish()

            # Put the processed data into lists
            self._task_design.append(task_obj.design)
            self._task_response.append(task_obj.response)
            self._task_bootstraps.append(task_obj.get_bootstraps())
            self._task_names.append(task_name)
            self._task_priors.append(task_obj.priors_data)

            regulators.append(task_obj.design.gene_names)
            targets.append(task_obj.response.gene_names)

            task_str = "Processing task #{tid} [{t}] complete [{sh} & {sh2}]"
            Debug.vprint(task_str.format(tid=task_id, t=task_name, sh=task_obj.design.shape,
                                         sh2=task_obj.response.shape), level=1)

        """
        task1_homologs = self.homologs[0]
        task2_homologs = self.homologs[1]
        self._task_design[0][self._task_design[0].gene_data]
        
        keep_column_bool = np.ones((len(self._task_design[0]._adata.var_names),), dtype=bool)
        
        has_homolog = self._task_response[0]._adata.var_names.isin(task1_homologs)
        
        if trim_gene_list is not None:
            keep_column_bool &= self._adata.var_names.isin(trim_gene_list)
        if "trim_gene_list" in self._adata.uns:
            keep_column_bool &= self._adata.var_names.isin(self._adata.uns["trim_gene_list"])

        list_trim = len(self._adata.var_names) - np.sum(keep_column_bool)
        comp = 0 if self._is_integer else np.finfo(self.values.dtype).eps * 10
        
        test  = AnnData(self._task_response[0]._adata.X[:, has_homolog],
                        obs = self._task_response[0]._adata.obs.copy(),
                        var = self._task_response[0]._adata.var.loc[has_homolog, :].copy(),
                        dtype = self._task_response[0]._adata.X.dtype)
        
        test2 = AnnData(self._task_response[0]._adata.X[:, ~has_homolog],
                        obs = self._task_response[0]._adata.obs.copy(),
                        var = self._task_response[0]._adata.var.loc[~has_homolog, :].copy(),
                        dtype = self._task_response[0]._adata.X.dtype)
        
        self._adata = AnnData(self._adata.X[:, keep_column_bool],
                                  obs=self._adata.obs.copy(),
                                  var=self._adata.var.loc[keep_column_bool, :].copy(),
                                  dtype=self._adata.X.dtype)

            # Make sure that there's no hanging reference to the original object
        gc.collect()
        
        
        
        
        
        
        self._targets = amusr_regression.filter_genes_on_tasks(targets, self._target_expression_filter)
        self._regulators = amusr_regression.filter_genes_on_tasks(regulators, self._regulator_expression_filter)
        """        
        #self._targets = ms_amusr_regression.filter_genes_on_tasks(targets, self._target_expression_filter)
        #self._regulators = ms_amusr_regression.filter_genes_on_tasks(regulators, self._regulator_expression_filter)
        
        self._targets = targets
        self._regulators = regulators
        
        Debug.vprint("Processed data into design/response [{g} x {k}]".format(g=len(self._targets),
                                                                              k=len(self._regulators)), level=0)

        #Adding the gold standard from each task into the self.gold_standard objects as a list of length(_n_tasks)
        self.gold_standard = []
        for k in range(self._n_tasks):
            self.gold_standard.append(self._task_objects[k].gold_standard)
        # Clean up the TaskData objects and force a cyclic collection
        del self._task_objects
        gc.collect() # Something wierd here, design/response is now nested list

        # Make sure that the task data files have the correct columns
        
        for d in self._task_design:
            d.trim_genes(remove_constant_genes=True)
            #d.trim_genes(remove_constant_genes=False, trim_gene_list=self._regulators)

        for r in self._task_response:
            r.trim_genes(remove_constant_genes=True)
            #r.trim_genes(remove_constant_genes=False, trim_gene_list=self._targets)
        
        #Clean up homology list
        original_length = len(self.homologs)
        for k in range(self._n_tasks):
            #keep_list = self.homologs.iloc[:,k].isin(list(self._task_response[k].gene_data.index))
            keep_list = self.homologs.iloc[:,k].isin(list(self._task_response[k].gene_data.index))
            self.homologs = self.homologs.loc[keep_list]
        Debug.vprint("Removed {x} missing homolog pairs from the homology list".format(x=(original_length-len(self.homologs)), level=0))
        

    def emit_results(self, betas, rescaled_betas, gold_standard, priors_data):
        """
        Output result report(s) for workflow run.

        This is called when `.startup()` is run. It is not necessary to call separately.
        """
        if self.is_master():
            self.create_output_dir()
            rp = self._result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method,
                                               metric=self.metric)
            rp.tasks_names = self._task_names
            self.results = rp.summarize_network(self.output_dir, gold_standard, self._task_priors)
            self.task_results = rp.tasks_networks
            return self.results
        else:
            return None
        
    def create_tf_homology(self):
        1+1
        
        return
