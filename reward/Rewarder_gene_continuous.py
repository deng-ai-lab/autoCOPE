from architecture.preprocessing_scheme import PreprocessingScheme
import numpy as np
import scanpy as sc
import random
import torch
#import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

split_seed = 0

class Rewarder(object):
    def __init__(self, actions_p, actions_log_p, actions_index, len_std, args):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.actions_index = actions_index
        self.len_std = len_std

        self.args = args

        self.cos_score_val = None
        self.pear_score_val = None
        self.spearman_score_val = None
        self.r2_val = None
        self.mse_val = None
        self.reward = None
        self.len_score = None
        self.seed = 0


def metric(gt, pred):
    cos_sims = []
    pear_corrs = []
    spearman_corrs = []

    for v1, v2 in zip(gt.T, pred.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sim = (v1 @ v2) / norm_sq

        temp = pd.DataFrame(np.concatenate([v1.reshape([-1, 1]), v2.reshape([-1, 1])], axis=1), columns=['rc', 'gt'])
        pear_corr = temp.corr().values[0, 1]
        spearman_corr = temp.corr('spearman').values[0, 1]

        cos_sims.append(cos_sim)
        pear_corrs.append(pear_corr)
        spearman_corrs.append(spearman_corr)

    cos_sim = np.mean(np.array(cos_sims))
    pear_corr = np.mean(np.array(pear_corrs))
    spearman_corr = np.mean(np.array(spearman_corrs))

    r2 = r2_score(gt.T, pred.T)
    mse = mean_squared_error(gt.T, pred.T)

    return cos_sim, pear_corr, spearman_corr, r2, mse

def split(ad_ref, ad_target):
    gene_intersection = ad_ref.var_names.intersection(ad_target.var_names).values.tolist()
    if len(gene_intersection) < 6:
        return ad_ref, ad_target, False
    gene_index = [i in gene_intersection for i in ad_target.var_names.values.tolist()]
    ad_target = ad_target[:, gene_index]

    gene_index = [i in gene_intersection for i in ad_ref.var_names.values.tolist()]
    gene_unique = ad_ref[:, ~np.array(gene_index)].var_names.values.tolist()
    gene_train, gene_test = train_test_split(gene_intersection, test_size=0.2, random_state=split_seed)

    n_fold = 5
    skf = KFold(n_splits=n_fold, shuffle=True, random_state=split_seed)
    Train_Gene_Set = {}
    Valid_Gene_Set = {}
    for i, (train_index, valid_index) in enumerate(skf.split(gene_train)):
        #print(len(train_index), len(valid_index))
        Train_Gene_Set['Fold' + str(i)] = np.array(gene_train)[train_index].tolist()
        Valid_Gene_Set['Fold' + str(i)] = np.array(gene_train)[valid_index].tolist()

    ad_target.uns['Train_Gene_Set'] = Train_Gene_Set
    ad_target.uns['Valid_Gene_Set'] = Valid_Gene_Set
    ad_target.uns['Test_Gene_Set'] = gene_test

    ad_ref.uns['Train_Gene_Set'] = Train_Gene_Set
    ad_ref.uns['Valid_Gene_Set'] = Valid_Gene_Set
    ad_ref.uns['Test_Gene_Set'] = gene_test
    ad_ref.uns['Unique_Gene_Set'] = gene_unique

    return ad_ref, ad_target, True

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


    # 1. Read target data and reference data and finish preprocessing.
    ad_target = sc.read(rewarder.args.Hdf5Path_target)
    ad_reference = sc.read(rewarder.args.Hdf5Path_ref)
    genes = ad_reference.var_names.values.tolist()

    graph = PreprocessingScheme(rewarder.args, rewarder.actions_index,
                                       rewarder.len_std)
    data, gene_id = graph.compute(x = ad_reference.X.toarray()
                                , x_gene_id = genes)
    if not output_valid_check(data):
        print('')
        print('Invalid Preprocessing Scheme is detected!!!')
        rewarder.cos_score_val = rewarder.args.nan_penalty
        rewarder.pear_score_val = rewarder.args.nan_penalty
        rewarder.spearman_score_val = rewarder.args.nan_penalty
        rewarder.r2_val = rewarder.args.nan_penalty
        rewarder.mse_val = rewarder.args.nan_penalty
        rewarder.len_score = rewarder.args.nan_penalty
        rewarder.reward = rewarder.args.nan_penalty
        return
    else:
        ad_ref = sc.AnnData(data)
        ad_ref.var_names = gene_id
        ad_ref.obs_names = ad_reference.obs_names
        ad_ref.obs = ad_reference.obs
        ad_reference, ad_target, flag = split(ad_ref, ad_target)
        if flag == False:
            print('')
            print('Invalid Preprocessing Scheme is detected!!!')
            rewarder.cos_score_val = rewarder.args.nan_penalty
            rewarder.pear_score_val = rewarder.args.nan_penalty
            rewarder.spearman_score_val = rewarder.args.nan_penalty
            rewarder.r2_val = rewarder.args.nan_penalty
            rewarder.mse_val = rewarder.args.nan_penalty
            rewarder.len_score = rewarder.args.nan_penalty
            rewarder.reward = rewarder.args.nan_penalty
            return

    # 2. Execute 5-fold validation on the training set and get the metrics for computing the reward value.
    n_fold = 5
    cos_sims, pear_corrs, spearman_corrs, r2s, mses = [], [], [], [], []
    for i in range(n_fold):
        ad_reference_train = ad_reference[:, ad_reference.uns['Train_Gene_Set']['Fold' + str(i)]]
        ad_target_train = ad_target[:, ad_target.uns['Train_Gene_Set']['Fold' + str(i)]]

        try:
            clf = LinearRegression()
            clf.fit(ad_reference_train.X.toarray().transpose(), ad_target_train.X.toarray().transpose())
        except:
            print('')
            print('Invalid Preprocessing Scheme is detected!!!')
            rewarder.cos_score_val = rewarder.args.nan_penalty
            rewarder.pear_score_val = rewarder.args.nan_penalty
            rewarder.spearman_score_val = rewarder.args.nan_penalty
            rewarder.r2_val = rewarder.args.nan_penalty
            rewarder.mse_val = rewarder.args.nan_penalty
            rewarder.len_score = rewarder.args.nan_penalty
            rewarder.reward = rewarder.args.nan_penalty
            return

        ad_reference_valid = ad_reference[:, ad_reference.uns['Valid_Gene_Set']['Fold' + str(i)]]
        ad_target_valid_predict = clf.predict(ad_reference_valid.X.toarray().transpose()).transpose()
        ad_target_valid_gt = ad_target[:, ad_target.uns['Valid_Gene_Set']['Fold' + str(i)]].X.toarray()

        cos_sim, pear_corr, spearman_corr, r2, mse = metric(ad_target_valid_gt, ad_target_valid_predict)
        cos_sims.append(cos_sim)
        pear_corrs.append(pear_corr)
        spearman_corrs.append(spearman_corr)
        r2s.append(r2)
        mses.append(mse)
    cos_sim_valid = np.mean(np.array(cos_sims))
    pear_corr_valid = np.mean(np.array(pear_corrs))
    spearman_corr_valid = np.mean(np.array(spearman_corrs))
    r2_valid = np.mean(np.array(r2s))
    mse_valid = np.mean(np.array(mses))
    if np.isnan(cos_sim_valid):
        rewarder.cos_score_val = rewarder.args.nan_penalty
    else:
        rewarder.cos_score_val = cos_sim_valid
    if np.isnan(pear_corr_valid):
        rewarder.pear_score_val = rewarder.args.nan_penalty
    else:
        rewarder.pear_score_val = pear_corr_valid
    if np.isnan(spearman_corr_valid):
        rewarder.spearman_score_val = rewarder.args.nan_penalty
    else:
        rewarder.spearman_score_val = spearman_corr_valid
    if np.isnan(r2_valid):
        rewarder.r2_val = rewarder.args.nan_penalty
    else:
        rewarder.r2_val = r2_valid
    if np.isnan(mse_valid):
        rewarder.mse_val = rewarder.args.nan_penalty
    else:
        rewarder.mse_val = mse_valid


    # 3. Reward the preprocessed data.
    rewarder.len_score = (rewarder.args.graph_max_size - rewarder.len_std[0]) / rewarder.args.graph_max_size * rewarder.args.graph_length_factor
    rewarder.reward = rewarder.cos_score_val + rewarder.len_score