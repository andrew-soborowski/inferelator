import pandas as pd
import numpy as np
import os

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing import results_processor
from inferelator.postprocessing import BETA_SIGN_COLUMN, MEDIAN_EXPLAIN_VAR_COLUMN
from inferelator.postprocessing.results_processor_mtl import ResultsProcessorMultiTask, _df_resizer

class ResultsProcessorMultiSpecies(ResultsProcessorMultiTask):
    """
    This results processor should handle the results of the MultiTask inferelator

    It will output the results for each task, as well as rank-combining to construct a network from all tasks
    """
    def summarize_network(self, output_dir, gold_standard, priors):
        """
        Take the betas and rescaled beta_errors, construct a network, and test it against the gold standard
        :param output_dir: str
            Path to write files into. Don't write anything if this is None.
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard to test the network against
        :param priors: list(pd.DataFrame [G x K])
            Prior data
        :return overall_result: InferelatorResult
            Returns an InferelatorResult for the final aggregate network
        """

        assert check.argument_type(priors, list)
        assert len(priors) == len(self.tasks_names)
        assert len(self.betas) == len(self.tasks_names)
        assert len(self.rescaled_betas) == len(self.tasks_names)

        # Create empty lists for task-specific results
        overall_confidences = []
        overall_resc_betas = []

        # Get intersection of indices
        gene_set = list(set([i for df in self.betas for i in df[0].index.tolist()]))
        tf_set = list(set([i for df in self.betas for i in df[0].columns.tolist()]))

        # Use the existing indices if there's no difference from the intersection
        gene_set = gene_set if len(self.betas[0][0].index.symmetric_difference(gene_set)) != 0 else self.betas[0][0].index
        tf_set = tf_set if len(self.betas[0][0].columns.symmetric_difference(tf_set)) != 0 else self.betas[0][0].columns

        # Create empty dataframes for task-specific results
        overall_sign = pd.DataFrame(np.zeros((len(gene_set), len(tf_set))),
                                    index=gene_set,
                                    columns=tf_set)

        overall_threshold = overall_sign.copy()

        if not isinstance(gold_standard, list):
            gold_standard = [gold_standard] * len(self.tasks_names)

        # Run the result processing on each individual task
        # Keep the confidences from the tasks so that a final aggregate network can be assembled
        # Store the individual task network results in a dict
        self.tasks_networks = {}
        
        for task_id, task_name in enumerate(self.tasks_names):
            task_rs_calc = self.metric(self.rescaled_betas[task_id], gold_standard[task_id],
                                       filter_method=self.filter_method)

            task_threshold, task_sign, task_nonzero = self.threshold_and_summarize(self.betas[task_id], self.threshold)
            task_resc_betas_mean, task_resc_betas_median = self.mean_and_median(self.rescaled_betas[task_id])

            task_extra_cols = {BETA_SIGN_COLUMN: task_sign, MEDIAN_EXPLAIN_VAR_COLUMN: task_resc_betas_median}

            task_network_data = self.process_network(task_rs_calc, priors[task_id], beta_threshold=task_threshold,
                                                     extra_columns=task_extra_cols)

            # Pile up data
            overall_confidences.append(_df_resizer(task_rs_calc.all_confidences, gene_set, tf_set))
            overall_resc_betas.append(_df_resizer(task_resc_betas_median, gene_set, tf_set))
            overall_sign += np.sign(_df_resizer(task_sign, gene_set, tf_set))
            overall_threshold += _df_resizer(task_threshold, gene_set, tf_set)

            m_name, score = task_rs_calc.score()
            utils.Debug.vprint("Task {t} Model {m}:\t{score}".format(t=task_name, m=m_name, score=score), level=0)

            task_result = self.result_object(task_network_data, task_threshold, task_rs_calc.all_confidences,
                                             task_rs_calc, betas_sign=task_sign, betas=self.betas[task_id])

            if self.write_task_files is True and output_dir is not None:
                task_result.write_result_files(os.path.join(output_dir, task_name))

            self.tasks_networks[task_id] = task_result
        
        
        combined_gold_standard = self.combine_gold_standard(gold_standard)
        overall_rs_calc = self.metric(overall_confidences, combined_gold_standard, filter_method=self.filter_method)

        overall_threshold = (overall_threshold / len(overall_confidences) > self.threshold).astype(int)
        overall_resc_betas_mean, overall_resc_betas_median = self.mean_and_median(overall_resc_betas)
        extra_cols = {BETA_SIGN_COLUMN: overall_sign, MEDIAN_EXPLAIN_VAR_COLUMN: overall_resc_betas_median}

        m_name, score = overall_rs_calc.score()
        utils.Debug.vprint("Aggregate Model {m}:\t{score}".format(m=m_name, score=score), level=0)

        network_data = self.process_network(overall_rs_calc, None, beta_threshold=overall_threshold,
                                            extra_columns=extra_cols)

        overall_result = self.result_object(network_data, overall_threshold,
                                            _df_resizer(overall_rs_calc.all_confidences, gene_set, tf_set),
                                            overall_rs_calc, betas_sign=overall_sign)
        overall_result.write_result_files(output_dir)
        overall_result.tasks = self.tasks_networks

        return overall_result
    
    #Creates a master gold standard by combining each of the individual gold standard files into 1
    #Assumes that each gold standard is completely unique in gene identifiers!
    @staticmethod
    def combine_gold_standard(gold_standard):
        targets, regulators = [], []
        for k in range(len(gold_standard)):
            targets.append(list(gold_standard[k].index))
            regulators.append(list(gold_standard[k].columns))
        targets = [y for x in targets for y in x]
        regulators = [y for x in regulators for y in x]
        combined_gold_standard = pd.DataFrame(0,index = targets, columns = regulators)
        for k in range(len(gold_standard)):
            combined_gold_standard = combined_gold_standard.add(gold_standard[0], fill_value=0)
        return combined_gold_standard