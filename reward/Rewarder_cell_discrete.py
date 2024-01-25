from architecture.preprocessing_scheme import PreprocessingScheme
import torch
from sklearn.neighbors import NearestCentroid
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics import classification_report
#import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import random


class Rewarder(object):
    def __init__(self, actions_p, actions_log_p, actions_index, len_std, args):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.actions_index = actions_index
        self.len_std = len_std

        self.args = args
        self.device = self.args.gpu
        self.metric = 'medianF1'

        self.f1 = None
        self.len_score = None
        self.reward = None
        self.seed = 0


def get_score(rewarder):
    torch.manual_seed(rewarder.seed)
    np.random.seed(rewarder.seed)
    random.seed(rewarder.seed)
    #tf.random.set_seed(rewarder.seed)
    ss_1 = StratifiedKFold(n_splits=5)
    
    ss_3 = StratifiedShuffleSplit(n_splits=5, test_size=0.20, train_size=0.80, random_state=rewarder.seed)

    # read and preprocess the data
    ad_ = sc.read(rewarder.args.Hdf5Path_ref)
    x_cell_id = list(ad_.obs_names.values)
    x_gene_id = list(ad_.var_names.values)
    cell_nums = len(x_cell_id)

    graph = PreprocessingScheme(rewarder.args, rewarder.actions_index, rewarder.len_std)
    data, gene_id = graph.compute(x=ad_.X.toarray()
                                  , x_gene_id=x_gene_id
                                  , y=None
                                  , y_gene_id=None)

    if not isinstance(data, np.ndarray):
        if data == None:
            print('')
            print('Invalid Preprocessing Scheme is detected!!!')
            rewarder.f1 = rewarder.args.nan_penalty
            rewarder.reward = rewarder.args.nan_penalty
            rewarder.len_score = rewarder.args.nan_penalty
            return
    labels = ad_.obs['region_type'].values

    train_val_cell_indexes = []
    test_cell_indexes = []
    for train_index, test_index in ss_1.split(data, np.array(labels)):
        train_val_cell_indexes.append(train_index)
        test_cell_indexes.append(test_index)
    train_val_cell_index = train_val_cell_indexes[0]
    for train_index, test_index in ss_3.split(data[train_val_cell_index, :], np.array(labels)[train_val_cell_index]):
        train_index = train_index
        val_index = test_index
        break

    train_data = data[train_val_cell_index, :][train_index, :]
    val_data = data[train_val_cell_index, :][val_index, :]
    #print(train_data.shape)
    #print(val_data.shape)
    y_train = np.array(np.array(labels)[train_val_cell_index][train_index])
    y_val = np.array(np.array(labels)[train_val_cell_index][val_index])

    train_gene_id = val_gene_id = gene_id

    train = pd.DataFrame(data=train_data, columns=train_gene_id)
    val = pd.DataFrame(data=val_data, columns=val_gene_id)
    #print(train)
    #print(val)

    cell_type_name = list(set(list(ad_.obs['region_type'])))

    Classifier = NearestCentroid()

    truelab = []
    pred = []
    try:
        Classifier.fit(train, y_train)

        predicted = Classifier.predict(val)
    except:
        print('')
        print('Invalid data for classifier')
        rewarder.f1 = rewarder.args.nan_penalty
        rewarder.reward = rewarder.args.nan_penalty
        rewarder.len_score = rewarder.args.nan_penalty
        return

    truelab.extend(y_val)
    pred.extend(predicted)

    truelab = pd.DataFrame(truelab)
    pred = pd.DataFrame(pred)

    report = classification_report(truelab, pred, output_dict=True, target_names=cell_type_name)
    df = pd.DataFrame(report).transpose()
    mean_f1 = np.median(df['f1-score'][:-3])

    print('valid acc {:.4f}'.format(mean_f1))

    rewarder.f1 = mean_f1
    num_modality = 1
    rewarder.len_score = (num_modality * rewarder.args.graph_max_size - rewarder.len_std[0]) / rewarder.args.graph_max_size / num_modality * rewarder.args.graph_length_factor
    rewarder.reward = rewarder.f1 + rewarder.len_score