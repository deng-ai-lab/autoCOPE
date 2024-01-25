from architecture.preprocessing_scheme import PreprocessingScheme
import numpy as np
import scanpy as sc
import random
import torch
#import tensorflow as tf
import anndata as ad
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import pandas as pd

def annotate_gene_sparsity(adata):
    """
    Annotates gene sparsity in given Anndatas.
    Update given Anndata by creating `var` "sparsity" field with gene_sparsity (1 - % non-zero observations).

    Args:
        adata (Anndata): single cell or spatial data.

    Returns:
        None
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity


def assess_spatial_trans_reconstruction(adata_st_rc, adata_st_gt, adata_sc=None):
    """ Assesses reconstructed spatial data with the true spatial data

    Args:
        adata_st_rc (AnnData): generated spatial data returned by `project_genes`
        adata_st_gt (AnnData): gene spatial data
        adata_sc (AnnData): Optional. When passed, sparsity difference between adata_sc and adata_sp will be calculated. Default is None.

    Returns:
        Pandas Dataframe: a dataframe with columns: 'score', 'sparsity_sp'(spatial data sparsity).
                          Columns - 'sparsity_sc'(single cell data sparsity), 'sparsity_diff'(spatial sparsity - single cell sparsity) returned only when adata_sc is passed.
    """

    overlap_genes = adata_st_rc.var_names.values.tolist()

    annotate_gene_sparsity(adata_st_gt)

    # Annotate cosine similarity of each training gene
    cos_sims = []
    pear_corrs = []
    spearman_corrs = []

    if hasattr(adata_st_rc.X, "toarray"):
        X_1 = adata_st_rc[:, overlap_genes].X.toarray()
    else:
        X_1 = adata_st_rc[:, overlap_genes].X
    if hasattr(adata_st_gt.X, "toarray"):
        X_2 = adata_st_gt[:, overlap_genes].X.toarray()
    else:
        X_2 = adata_st_gt[:, overlap_genes].X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sim = (v1 @ v2) / norm_sq

        temp = pd.DataFrame(np.concatenate([v1.reshape([-1, 1]), v2.reshape([-1, 1])], axis=1), columns=['rc', 'gt'])
        pear_corr = temp.corr().values[0, 1]
        spearman_corr = temp.corr('spearman').values[0, 1]

        cos_sims.append(cos_sim)
        pear_corrs.append(pear_corr)
        spearman_corrs.append(spearman_corr)

    df_g = pd.DataFrame(np.array([cos_sims, pear_corrs, spearman_corrs]).transpose(), overlap_genes,
                        columns=["cos_score", "pear_score", "spearman_score"])
    df_g["sparsity_st"] = adata_st_gt[:, overlap_genes].var.sparsity

    df_g = df_g.sort_values(by="cos_score", ascending=False)
    return df_g

class Rewarder(object):
    def __init__(self, actions_p, actions_log_p, actions_index, len_std, args):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.actions_index = actions_index
        self.len_std = len_std

        self.args = args

        self.mse = None
        self.r2 = None
        self.spear = None
        self.spearman = None
        self.cos = None
        self.reward = None
        self.len_score = None
        self.seed = 0


def TE_get_splits(XT, n):
    skf = KFold(n_splits=n, random_state=0, shuffle=True)
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in skf.split(X=XT)]
    return ind_dict

def output_valid_check(data):
    if not isinstance(data, np.ndarray):
        if data == None:
            return False
        else:
            return True
    else:
        return True

def get_score(rewarder):
    # 0. Initialize the random seed to make sure reproducibility, only valid when multiprocessing mode is on.
    torch.manual_seed(rewarder.seed)
    np.random.seed(rewarder.seed)
    random.seed(rewarder.seed)
    #tf.random.set_seed(rewarder.seed)

    # 1. Read target data and reference data on intersected genes and finish preprocessing.
    ad_ = sc.read(rewarder.args.Hdf5Path_ref)
    ad_tar = sc.read(rewarder.args.Hdf5Path_target)

    try:
        data_tar = ad_tar.X.toarray()
    except:
        data_tar = ad_tar.X

    genes = [i for i in range(ad_.X.shape[1])]
    tar_genes = [i for i in range(ad_tar.X.shape[1])]

    graph = PreprocessingScheme(rewarder.args, rewarder.actions_index,
                                       rewarder.len_std)
    try:
        data, gene_id = graph.compute(x = ad_.X.toarray()
                                , x_gene_id = genes)
    except:
        data, gene_id = graph.compute(x=ad_.X
                                      , x_gene_id=genes)
    if not output_valid_check(data):
        print('')
        print('Invalid Preprocessing Scheme is detected (Type 1)!!!')
        rewarder.mse = rewarder.args.nan_penalty
        rewarder.r2 = rewarder.args.nan_penalty
        rewarder.spear = rewarder.args.nan_penalty
        rewarder.spearman = rewarder.args.nan_penalty
        rewarder.cos = rewarder.args.nan_penalty
        rewarder.len_score = rewarder.args.nan_penalty
        rewarder.reward = rewarder.args.nan_penalty
        return

    if rewarder.args.regressor == 'PLSR':
        model = PLSRegression()
    elif rewarder.args.regressor == 'EN':
        model = ElasticNet(random_state=rewarder.seed)
    elif rewarder.args.regressor == 'KNN':
        model = KNeighborsRegressor(n_neighbors=20)
        
    x_train_val, x_test, y_train_val, y_test = train_test_split(data, data_tar, train_size=0.80, random_state=0)
    
    n_cvfold = 5
    ind_dict_train_valid = TE_get_splits(x_train_val, n_cvfold)

    total_mse_val = []
    total_r2_val = []
    total_spear_val = []
    total_spearman_val = []
    total_cos_val = []
    for n in range(n_cvfold):
        train_ind = ind_dict_train_valid[n]['train']
        val_ind = ind_dict_train_valid[n]['val']
        train_data = x_train_val[train_ind, :]
        y_train = y_train_val[train_ind, :]
        val_data = x_train_val[val_ind, :]
        y_val = y_train_val[val_ind, :]

        try:
            model.fit(train_data, y_train)
        except:
            print('')
            print('Invalid Preprocessing Scheme is detected (Type 2)!!!')
            rewarder.mse = rewarder.args.nan_penalty
            rewarder.r2 = rewarder.args.nan_penalty
            rewarder.spear = rewarder.args.nan_penalty
            rewarder.spearman = rewarder.args.nan_penalty
            rewarder.cos = rewarder.args.nan_penalty
            rewarder.len_score = rewarder.args.nan_penalty
            rewarder.reward = rewarder.args.nan_penalty
            return

        y_val_rc = model.predict(val_data)

        ad_st_rc = ad.AnnData(pd.DataFrame(data=y_val_rc, columns=tar_genes))
        ad_st_gt = ad.AnnData(pd.DataFrame(data=y_val, columns=tar_genes))
        df_g = assess_spatial_trans_reconstruction(ad_st_rc, ad_st_gt, ad_tar)

        cos_score = np.mean(df_g['cos_score'])
        pear_score = np.mean(df_g['pear_score'])
        spearman_score = np.mean(df_g['spearman_score'])
        MSE = np.mean((y_val_rc - y_val) * (y_val_rc - y_val))
        R2 = r2_score(y_val, y_val_rc)
        if np.isnan(cos_score) or np.isnan(pear_score) or np.isnan(spearman_score) or np.isnan(MSE) or np.isnan(R2):
            print('')
            print('Invalid Preprocessing Scheme is detected (Type 3)!!!')
            rewarder.mse = rewarder.args.nan_penalty
            rewarder.r2 = rewarder.args.nan_penalty
            rewarder.spear = rewarder.args.nan_penalty
            rewarder.spearman = rewarder.args.nan_penalty
            rewarder.cos = rewarder.args.nan_penalty
            rewarder.len_score = rewarder.args.nan_penalty
            rewarder.reward = rewarder.args.nan_penalty
            return

        total_mse_val.append(MSE)
        total_r2_val.append(R2)
        total_spear_val.append(pear_score)
        total_spearman_val.append(spearman_score)
        total_cos_val.append(cos_score)
    rewarder.mse = np.mean(total_mse_val)
    rewarder.r2 = np.mean(total_r2_val)
    rewarder.spear = np.mean(total_spear_val)
    rewarder.spearman = np.mean(total_spearman_val)
    rewarder.cos = np.mean(total_cos_val)
    rewarder.len_score = (rewarder.args.graph_max_size - rewarder.len_std[0]) / rewarder.args.graph_max_size * rewarder.args.graph_length_factor
    rewarder.reward = np.exp(-rewarder.mse) + rewarder.len_score