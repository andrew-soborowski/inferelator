#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:05:37 2020

@author: andrew
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.special import comb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils
from inferelator import default
from inferelator.regression import base_regression
from inferelator.regression.bbsr_python import BBSR, BBSRRegressionWorkflowMixin
import shelve
# Shadow built-in zip with itertools.izip if this is python2 (This puts out a memory dumpster fire)
try:
    from itertools import izip as zip
except ImportError:
    pass

DEFAULT_prior_weight = 1


class AMuSR_OneGene:

    max_iter = 1000
    tolerance = 1e-2

    def __init__(self, n_tasks, n_features):

        self.n_tasks = n_tasks
        self.n_features = n_features


    def preprocess_data(self, X, Y):
        """
        center and standardize input X and Y data (z-score)
        """

        for k in range(self.n_tasks):
            X[k] = StandardScaler().fit_transform(X[k].astype(float))
            Y[k] = StandardScaler().fit_transform(Y[k].astype(float))

        return((X, Y))

    def covariance_update_terms(self, X, Y):
        """
        returns C and D, containing terms for covariance update for OLS fit
        C: transpose(X_j)*Y for each feature j
        D: transpose(X_j)*X_l for each feature j for each feature l
        reference: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
        Regularization Paths for Generalized Linear Models via Coordinate Descent.
        """

        C = np.zeros((self.n_tasks, self.n_features))
        D = np.zeros((self.n_tasks, self.n_features, self.n_features))

        for k in range(self.n_tasks):
            C[k] = np.dot(Y[k].transpose(), X[k])
            D[k] = np.dot(X[k].transpose(), X[k])

        return((C, D))


    def updateS(self, C, D, B, S, lamS, prior):
        """
        returns updated coefficients for S (predictors x tasks)
        lasso regularized -- using cyclical coordinate descent and
        soft-thresholding
        """
        # update each task independently (shared penalty only)
        for k in range(self.n_tasks):
            c = C[k]; d = D[k]
            b = B[:,k]; s = S[:,k]
            p = prior[:,k]
            # cycle through predictors
            for j in range(self.n_features):
                # set sparse coefficient for predictor j to zero
                s_tmp = deepcopy(s)
                s_tmp[j] = 0.
                # calculate next coefficient based on fit only
                if d[j,j] == 0:
                    alpha = 0
                else:
                    alpha = (c[j]-np.sum((b+s_tmp)*d[j]))/d[j,j]
                # lasso regularization
                if abs(alpha) <= p[j]*lamS:
                    s[j] = 0.
                else:
                    s[j] = alpha-(np.sign(alpha)*p[j]*lamS)
            # update current task
            S[:,k] = s

        return(S)

    def updateB(self, C, D, B, S, lamB, prior):
        """
        returns updated coefficients for B (predictors x tasks)
        block regularized (l_1/l_inf) -- using cyclical coordinate descent and
        soft-thresholding on the l_1 norm across tasks
        reference: Liu et al, ICML 2009. Blockwise coordinate descent procedures
        for the multi-task lasso, with applications to neural semantic basis discovery.
        """
        #p = prior.min(axis=1)
        # cycles through predictors
        for j in range(self.n_features):
            # initialize next coefficients
            alphas = np.zeros(self.n_tasks)
            # update tasks for each predictor together
            for k in range(self.n_tasks):
                # get task covariance update terms
                c = C[k]; d = D[k]
                # get previous block-sparse and sparse coefficients
                b = B[:,k]; s = S[:,k]
                # set block-sparse coefficient for feature j to zero
                b_tmp = deepcopy(b)
                b_tmp[j] = 0.
                # calculate next coefficient based on fit only
                if d[j,j] == 0:
                    alphas[k] = 0
                else:
                    alphas[k] = (c[j]-np.sum((b_tmp+s)*d[:,j]))/d[j,j]
            # set all tasks to zero if l1-norm less than lamB
            if np.linalg.norm(alphas, 1) <= lamB:
                B[j,:] = np.zeros(self.n_tasks)
            # regularized update for predictors with larger l1-norm
            else:
                # find number of coefficients that would make l1-norm greater than penalty
                indices = np.abs(alphas).argsort()[::-1]
                sorted_alphas = alphas[indices]
                m_star = np.argmax((np.abs(sorted_alphas).cumsum()-lamB)/(np.arange(self.n_tasks)+1))
                # initialize new weights
                new_weights = np.zeros(self.n_tasks)
                # keep small coefficients and regularize large ones (in above group)
                for k in range(self.n_tasks):
                    idx = indices[k]
                    if k > m_star:
                        new_weights[idx] = sorted_alphas[k]
                    else:
                        sign = np.sign(sorted_alphas[k])
                        update_term = np.sum(np.abs(sorted_alphas)[:m_star+1])-lamB
                        new_weights[idx] = (sign/(m_star+1))*update_term
                # update current predictor
                B[j,:] = new_weights

        return(B)

    def fit(self, X, Y, lamB=0., lamS=0., C=None, D=None, S=None, B=None, prior=None):
        """
        Fits regression model in which the weights matrix W (predictors x tasks)
        is decomposed in two components: B that captures block structure across tasks
        and S that allows for the differences.
        reference: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning.
        """
        # calculate covariance update terms if not provided
        if C is None or D is None:
            C, D = self.covariance_update_terms(X, Y)
        # if S and B are provided -- warm starts -- will run faster
        if S is None or B is None:
            S = np.zeros((self.n_features, self.n_tasks))
            B = np.zeros((self.n_features, self.n_tasks))
        if prior is None:
            prior = np.ones((self.n_features, self.n_tasks))
        # initialize W
        W = S + B
        for n_iter in range(self.max_iter):
            # save old values of W (to check convergence)
            W_old = deepcopy(W)
            # update S and B coefficients
            S = self.updateS(C, D, B, S, lamS, prior)
            B = self.updateB(C, D, B, S, lamB, prior)
            W = S + B
            # update convergence criteria
            update = np.max(np.abs(W-W_old))
            if update < self.tolerance:
                break
        # weights matrix (W) is the sum of a sparse (S) and a block-sparse (B) matrix
        W = S + B
        # set small values of W to zero
        # since we don't run the algorithm until update equals zero
        W[np.abs(W) < 0.1] = 0

        return(W, S, B)


class AMuSR_regression(base_regression.BaseRegression):

    X = None  # list(pd.DataFrame [N, K])
    Y = None  # list(pd.DataFrame [N, G])
    priors = None  # list(pd.DataFrame [G, K]) OR pd.DataFrame [G, K]
    tfs = None  # pd.Index OR list
    genes = None  # pd.Index OR list
    homologs = None  #list(pd.DataFrae[G, n_tasks])
    
    K = None  # int
    G = None  # int
    n_tasks = None  # int
    prior_weight = 1.0  # float
    remove_autoregulation = True  # bool

    def __init__(self, X, Y, tfs=None, genes=None, priors=None, homologs=None, prior_weight=1, remove_autoregulation=True):
        """
        Set up a regression object for multitask regression
        :param X: list(pd.DataFrame [N, K])
        :param Y: list(pd.DataFrame [N, G])
        :param priors: pd.DataFrame [G, K]
        :param prior_weight: float
        :param remove_autoregulation: bool
        """

        # Set the data into the regression object
        self.X = X
        self.Y = Y
        self.n_tasks = len(X)

        # Set the priors and weight into the regression object
        self.priors = priors
        self.prior_weight = float(prior_weight)
        self.homologs = homologs
        
        # Construct a list of TFs & genes if they are not passed in
        if tfs is None or genes is None:
            tfs, genes = [], []
            for design, response in zip(X, Y):
                tfs.append(design.columns)
                genes.append(response.columns)
            self.tfs = filter_genes_on_tasks(tfs, "intersection")
            self.genes = filter_genes_on_tasks(genes, "intersection")
        else:
            self.tfs, self.genes = tfs, genes

        # Set the regulators and targets into the regression object
        self.K, self.G = len(tfs), len(genes)
        self.remove_autoregulation = remove_autoregulation

    def format_weights(self, df, col, targets, regs):

        df[col] = pd.to_numeric(df[col])

        out = pd.pivot_table(df, index='target',
                             columns='regulator',
                             values=col,
                             fill_value=0.)

        out.columns.name = None
        out.index.name = None

        out = pd.concat([out,
                         pd.DataFrame(0., index=out.index,
                                      columns=np.setdiff1d(regs, out.columns))], axis=1)
        out = pd.concat([out,
                         pd.DataFrame(0., index=np.setdiff1d(targets, out.index),
                                      columns=out.columns)])
        out = out.loc[targets, regs]

        return out

    def regress(self):
        """
        Execute multitask (AMUSR)
        :return: list
            Returns a list of regression results that the amusr_regression pileup_data can process
        """

        if MPControl.is_dask():
            from inferelator.distributed.dask_functions import amusr_regress_dask
            return amusr_regress_dask(self.X, self.Y, self.priors, self.prior_weight, self.n_tasks, self.genes,
                                      self.tfs, self.G, remove_autoregulation=self.remove_autoregulation)

        def regression_maker(j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G),
                                 level=level)

            gene = self.genes[j]
            x, y, tasks, zero_tfs_list = [], [], [], []

            if self.remove_autoregulation:
                tfs = [t for t in self.tfs if t != gene]
            else:
                tfs = self.tfs

            for k in range(self.n_tasks):
                if gene in self.Y[k].gene_names:
                   # present_tfs, zero_tfs = self.X[k].find_missing_genes(tfs)
                    #zero_tfs_list.append(zero_tfs)
                    x.append(self.X[k].get_gene_data_ms(tfs))
                   # x.append(self.X[k].get_gene_data(tfs))  # list([N, K]) old
                    #x.append(self.X[k].get_gene_data(present_tfs)) # new
                    y.append(self.Y[k].get_gene_data(gene, force_dense=True).reshape(-1, 1))  # list([N, 1])
                    tasks.append(k)  # [T,]

            prior = format_prior(self.priors, gene, tasks, self.prior_weight, self.homologs)
            return run_regression_EBIC(x, y, tfs, tasks, gene, prior)
        regression_maker(0) #Testing line, remove later
        return MPControl.map(regression_maker, range(self.G))

    def pileup_data(self, run_data):

        weights = []
        rescaled_weights = []

        for k in range(self.n_tasks):
            results_k = []
            for res in run_data:
                try:
                    results_k.append(res[k])
                except KeyError:
                    pass

            results_k = pd.concat(results_k)
            weights_k = self.format_weights(results_k, 'weights', self.genes, self.tfs)
            rescaled_weights_k = self.format_weights(results_k, 'resc_weights', self.genes, self.tfs)
            rescaled_weights_k[rescaled_weights_k < 0.] = 0

            weights.append(weights_k)
            rescaled_weights.append(rescaled_weights_k)

        return weights, rescaled_weights


def sum_squared_errors(X, Y, W, k):
    '''
    Get RSS for a particular task 'k'
    '''
    return(np.sum((Y[k].T-np.dot(X[k], W[:,k]))**2))


def ebic(X, Y, W, n_tasks, n_samples, n_preds, gamma=1):
    '''
    Calculate EBIC for each task, and take the mean
    '''
    EBIC = []

    for k in range(n_tasks):

        n = n_samples[k]        
        nonzero_pred = (W[:,k] != 0).sum()
        RSS = sum_squared_errors(X, Y, W, k)
        BIC_penalty = nonzero_pred * np.log(n)
        BIC_extension = 2 * gamma * np.log(comb(n_preds, nonzero_pred))
        EBIC.append((n * np.log(RSS/n)) + BIC_penalty + BIC_extension)
        
    EBIC = np.mean(EBIC)

    return(EBIC)



def final_weights(X, y, TFs, gene):
    """
    returns reduction on variance explained for each predictor
    (model without each predictor compared to full model)
    see: Greenfield et al., 2013. Robust data-driven incorporation of prior
    knowledge into the inference of dynamic regulatory networks.
    """
    n_preds = len(TFs)
    # linear fit using sklearn
    ols = LinearRegression()
    ols.fit(X, y)
    # save weights and initialize rescaled weights vector
    weights = ols.coef_[0]
    resc_weights = np.zeros(n_preds)
    # variance of residuals (full model)
    var_full = np.var((y - ols.predict(X))**2)
    # when there is only one predictor
    if n_preds == 1:
        resc_weights[0] = 1 - (var_full/np.var(y))
    # remove each at a time and calculate variance explained
    else:
        for j in range(len(TFs)):
            X_noj = X[:, np.setdiff1d(range(n_preds), j)]
            ols = LinearRegression()
            ols.fit(X_noj, y)
            var_noj = np.var((y - ols.predict(X_noj))**2)
            resc_weights[j] = 1 - (var_full/var_noj)
    # format output
    out_weights = pd.DataFrame([TFs, [gene]*len(TFs), weights, resc_weights]).transpose()
    out_weights.columns = ['regulator', 'target', 'weights', 'resc_weights']
    return(out_weights)


def run_regression_EBIC(X, Y, TFs, tasks, gene, prior):
    """
    Run multitask regression
    :param X: list(np.ndarray [N x K]) [t]
        List consisting of design matrixes for each task
    :param Y: list(np.ndarray [N x 1]) [t]
        List consisting of response matrixes for each task
    :param TFs: list [K]
        List of TF names for each feature
    :param tasks: list(int) [t]
        List identifying each task
    :param gene: str
        The gene being modeled
    :param prior: np.ndarray [K x T]
        The priors for this gene in a TF x Task array
    :return: dict
    """

    assert len(X) == len(Y)
    assert len(X) == len(tasks)

    # The number of tasks
    n_tasks = len(X)

    # The number of predictors
    n_preds = X[0].shape[1]

    # A list of the number of samples for each task
    n_samples = [X[k].shape[0] for k in range(n_tasks)]

    ###### EBIC ######
    Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]
    Ss = np.linspace((1.0/n_tasks)+0.01, 0.99, 10)[::-1] # in paper I used 0.51 as minimum for all networks
    lamBparam = np.sqrt((n_tasks * np.log(n_preds))/np.mean(n_samples))

    model = AMuSR_OneGene(n_tasks, n_preds)
    X, Y = model.preprocess_data(X, Y)
    C, D = model.covariance_update_terms(X, Y)
    S = np.zeros((n_preds, n_tasks))
    B = np.zeros((n_preds, n_tasks))

    min_ebic = float('Inf')

    outW = None
    for c in Cs:
        tmp_lamB = c * lamBparam
        for s in Ss:
            tmp_lamS = s * tmp_lamB
            W, S, B = model.fit(X, Y, tmp_lamB, tmp_lamS, C, D, S, B, prior)
            ebic_score = ebic(X, Y, W, n_tasks, n_samples, n_preds)
            if ebic_score < min_ebic:
                min_ebic = ebic_score
                lamB = tmp_lamB
                lamS = tmp_lamS
                outW = W

    ###### RESCALE WEIGHTS ######
    output = {}

    if outW is not None:
        for kx, k in enumerate(tasks):
            nonzero = outW[:,kx] != 0
            if nonzero.sum() > 0:
                cTFs = np.asarray(TFs)[outW[:,kx] != 0]
                output[k] = final_weights(X[kx][:, nonzero], Y[kx], cTFs, gene)
    return(output)


# Edited to allow pulling the proper homolog out of the prior
def format_prior(priors, gene, tasks, prior_weight, homologs):
    '''
    Returns priors for one gene (numpy matrix TFs by tasks)
    '''
    if priors is None:
        return None

    if isinstance(priors, list):
        priors_out = []
        homologs = homologs[homologs[0].str.contains(gene)]
        for k in tasks:
            gene = homologs.iloc[0,k]
            prior = priors[k]
            prior = prior.loc[gene, :].replace(np.nan, 0)
            priors_out.append(weight_prior(prior, prior_weight))
        priors_out = np.transpose(np.asarray(priors_out))
    else:
        if gene in priors.index:
            priors_out = np.tile(weight_prior(priors.loc[gene, :].values, prior_weight).reshape(-1, 1),
                                 (1, len(tasks)))
        else:
            priors_out = np.zeros((priors.shape[1], len(tasks)))

    return priors_out


def weight_prior(prior, prior_weight):
    prior = (prior != 0).astype(float)
    prior /= prior_weight
    prior[prior == 0] = 1.0
    prior = prior / prior.sum() * len(prior)
    return prior


class MS_AMUSRRegressionWorkflowMixin(base_regression._MultitaskRegressionWorkflowMixin, BBSRRegressionWorkflowMixin):
    """
    Multi-Task AMuSR regression

    https://doi.org/10.1371/journal.pcbi.1006591
    """

    prior_weight = default.DEFAULT_prior_weight

    def set_regression_parameters(self, prior_weight=None):
        """
        Set regression parameters for AmUSR

        :param prior_weight: Weight for edges that are present in the prior network.
            Non-prior edges have a weight of 1. Set this to 1 to weight prior and non-prior edges equally.
            Defaults to 1.
        :type prior_weight: numeric
        """

        self._set_with_warning("prior_weight", prior_weight)

    def run_bootstrap(self, bootstrap_idx):
        read_betas_from_file = True
        if read_betas_from_file:
            betas, betas_resc = [],[]
            betas.append(pd.read_csv("Species_1_combined_Betas.csv", index_col=0))
            betas.append(pd.read_csv("Species_2_combined_Betas.csv", index_col=0))
            betas_resc.append(pd.read_csv("Species_1_combined_Betas_resc.csv", index_col=0))
            betas_resc.append(pd.read_csv("Species_2_combined_Betas_resc.csv", index_col=0))
            return betas, betas_resc
        x, y = [], []
        x_common, x_unique, y_common, y_unique = [], [], [], []
        
        self.create_homolog_tfs()
        
        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self._n_tasks):
            x.append(self._task_design[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx]))
            y.append(self._task_response[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx]))
            
            convert_names = False if k == 0 else True
            
            holder = y[k].separate_homolog_expression(self.homologs[k], convert_names, convert_names_to = self.homologs[0])
            y_common.append(holder[0])
            y_unique.append(holder[1])
            
            holder = x[k].separate_homolog_expression(self.tf_homologs[k], convert_names, convert_names_to = self.tf_homologs[0])
            x_common.append(holder[0])
            x_unique.append(holder[1])
        tfs, genes = [], []
        for design_common, design_unique, response in zip(x_common, x_unique, y_common):
            tfs.append(design_common.gene_data.index)
            tfs.append(design_unique.gene_data.index)
            genes.append(response.gene_data.index)
        tfs = filter_genes_on_tasks(tfs, "union")
        genes = filter_genes_on_tasks(genes, "intersection")
        homolog_priors = self.rebuild_priors(self._task_priors, tfs)
        MPControl.sync_processes(pref="amusr_pre")
        regress = AMuSR_regression(x, y_common, tfs=tfs, genes=genes, priors=homolog_priors,
                                   prior_weight=self.prior_weight, homologs = self.homologs)
        
        #regress = AMuSR_regression(x, y, tfs=self._regulators, genes=self._targets, priors=self._task_priors,
        #                           prior_weight=self.prior_weight)
        
        amusr_betas, amusr_betas_resc = regress.run()
        '''
        amusr_betas[0].to_csv("Species_1_amusr_Betas.csv")
        amusr_betas[1].to_csv("Species_2_amusr_Betas.csv")
        amusr_betas_resc[0].to_csv("Species_1_amusr_Betas_resc.csv")
        amusr_betas_resc[1].to_csv("Species_2_amusr_Betas_resc.csv")
        quit()
        '''
        bbsr_betas, bbsr_betas_resc = self.run_BBSR_bootstrap(x,y_unique)
        '''
        bbsr_betas, bbsr_betas_resc = [],[]
        bbsr_betas.append(pd.read_csv("Species_1_Betas.csv", index_col=0))
        bbsr_betas.append(pd.read_csv("Species_2_Betas.csv", index_col=0))
        bbsr_betas_resc.append(pd.read_csv("Species_1_Betas_resc.csv", index_col=0))
        bbsr_betas_resc.append(pd.read_csv("Species_2_Betas_resc.csv", index_col=0))
        amusr_betas, amusr_betas_resc = [],[]
        amusr_betas.append(pd.read_csv("Species_1_amusr_Betas.csv", index_col=0))
        amusr_betas.append(pd.read_csv("Species_2_amusr_Betas.csv", index_col=0))
        amusr_betas_resc.append(pd.read_csv("Species_1_amusr_Betas_resc.csv", index_col=0))
        amusr_betas_resc.append(pd.read_csv("Species_2_amusr_Betas_resc.csv", index_col=0))
        '''
        betas, betas_resc = [],[]
        for k in range(self._n_tasks):
            base_matrix = pd.DataFrame(data = np.zeros((len(self._targets[k]), len(self._regulators[k]))), index = self._targets[k].tolist(), columns = self._regulators[k].tolist())
            base_matrix_resc = base_matrix.copy()
            amusr_betas[k] = self.convert_and_trim_betas(amusr_betas[k], self.homologs, k)
            amusr_betas_resc[k] = self.convert_and_trim_betas(amusr_betas_resc[k], self.homologs, k)            
            
            base_matrix = base_matrix.add(bbsr_betas[k], fill_value=0)
            base_matrix = base_matrix.add(amusr_betas[k], fill_value=0)
            base_matrix_resc = base_matrix_resc.add(bbsr_betas_resc[k], fill_value=0)
            base_matrix_resc = base_matrix_resc.add(amusr_betas_resc[k], fill_value=0)
            betas.append(base_matrix)
            betas_resc.append(base_matrix_resc)
        
        return betas, betas_resc
        # Combine betas from each regression type
        # Return complete betas
        #return regress.run()
        
        
    # Creates a homolog list that only includes tfs where both genes are a tf
    # Needed because sometimes a tf will have a non-tf homolog, which should not get changed to its non-tf name
    # This is probably not the best place for this method but it works for now    
    def create_homolog_tfs(self):
        tf_homologs = self.homologs
        for k in range(self._n_tasks):
            tf_homologs = tf_homologs[tf_homologs[k].isin(self._regulators[k].to_list())]
        
        self.tf_homologs = tf_homologs
        return
    
    # Renames the tfs for every task except the first to match their homolog
    # Also, adds zero interaction between every tf not present and every gene
    def rebuild_priors(self, priors, tfs):
        for k in range(self._n_tasks):
            if k>0:
                converter = dict(pd.concat([self.tf_homologs[k], self.tf_homologs[0]], axis=1).values.tolist())
                names = list(priors[k].columns)
                for x in range(len(names)):
                    if names[x] in converter:
                        names[x] = converter[names[x]]
                #names = [converter[x] for x in names]
                priors[k].columns = names
            missing_tfs = list(set(tfs) - set(priors[k].columns))
            for new_tf in missing_tfs:
                priors[k][new_tf] = 0
        return priors


    def run_BBSR_bootstrap(self, X_list, Y_list):
         betas, betas_resc = [], []
    
        # Select the appropriate bootstrap from each task and stash the data into X and Y
         for k in range(self._n_tasks):
            #X = self._task_design[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx])
            #Y = self._task_response[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx])
            X = X_list[k]
            Y = Y_list[k]
    
            # Make sure that the priors align to the expression matrix
            priors_data = self._task_priors[k].reindex(labels=self._targets[k], axis=0). \
            reindex(labels=self._regulators[k], axis=1). \
            fillna(value=0)
            '''
            if self.clr_only:
                # Create a mock prior with no information if clr_only is set
                priors_data = pd.DataFrame(0, index=priors_data.index, columns=priors_data.columns)
            '''
            MPControl.sync_processes(pref="bbsr_pre")
            
           # Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=0)
            clr_matrix, _ = self.mi_driver().run(Y, X, return_mi=False)
            
           # Debug.vprint('Calculating task {k} betas using BBSR'.format(k=k), level=0)
            t_beta, t_br = BBSR(X, Y, clr_matrix, priors_data,
                                prior_weight=self.prior_weight, no_prior_weight=self.no_prior_weight,
                                nS=self.bsr_feature_num).run()
            betas.append(t_beta)
            betas_resc.append(t_br)
            
         return betas, betas_resc
     
    # Back converts the homolog converted row and column names for the amusr runs to return genes as 
    # those of their original species. Also, filters out TFs not found in the original species (these gene
    # would have no homolog)
    def convert_and_trim_betas(self, betas, homologs, k):
        converter = dict(pd.concat([self.tf_homologs[k], self.tf_homologs[0]], axis=1).values.tolist())
        col_names = list(betas.columns)
        new_col_names = []
        for name in col_names:
            if name in converter: 
                new_col_names.append(converter[name]) 
            else: 
                new_col_names.append(name)
        row_names = list(betas.index)
        new_row_names = []
        for name in row_names:
            if name in converter:
                new_row_names.append(converter[name])
            else:
                new_row_names.append(name)
        betas.columns = new_col_names
        betas.index = new_row_names
        keep_columns = list(set(new_col_names) & set(self._regulators[k]))
        betas_final = betas[keep_columns]
        return betas_final
    





def filter_genes_on_tasks(list_of_indexes, task_expression_filter):
    """
    Take a list of indexes and filter them based on the method specified in task_expression_filter to a single
    index
    :param list_of_indexes: list(pd.Index)
    :param task_expression_filter: str or int
    :return filtered_genes: pd.Index
    """

    filtered_genes = list_of_indexes[0]

    # If task_expression_filter is a number only keep genes in that number of tasks or higher
    if isinstance(task_expression_filter, int):
        filtered_genes = pd.concat(list(map(lambda x: x.to_series(), list_of_indexes))).value_counts()
        filtered_genes = filtered_genes[filtered_genes >= task_expression_filter].index
    # If task_expression_filter is "intersection" only keep genes in all tasks
    elif task_expression_filter == "intersection":
        for gene_idx in list_of_indexes:
            filtered_genes = filtered_genes.intersection(gene_idx)
    # If task_expression_filter is "union" keep genes that are in any task
    elif task_expression_filter == "union":
        for gene_idx in list_of_indexes:
            filtered_genes = filtered_genes.union(gene_idx)
    else:
        raise ValueError("{v} is not an allowed task_expression_filter value".format(v=task_expression_filter))

    return filtered_genes
