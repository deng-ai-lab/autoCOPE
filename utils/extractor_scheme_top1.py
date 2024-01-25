from architecture_utils.preprocessing_scheme_ import PreprocessingScheme
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import random
import torch
import tensorflow as tf

class Extractor(object):
    def __init__(self, dataset_path, graph_path, random_state, args):
        self.dataset_path = dataset_path
        self.graph_path = graph_path
        self.random_state = random_state
        self.save_path = args.save_path    #Path should be ended with '/'.

        self.args = args


def save_extracted_dataset(extractor):
    torch.manual_seed(extractor.args.seed_test)
    np.random.seed(extractor.args.seed_test)
    random.seed(extractor.args.seed_test)
    tf.random.set_seed(extractor.args.seed_test)

    # read and preprocess the data
    ad_ = sc.read(extractor.dataset_path)
    data, gene_id = ad_.X.toarray(), list(ad_.var_names.values)

    graph = PreprocessingScheme(extractor.args, [None, np.load(extractor.graph_path, allow_pickle=True)], None)
    data, gene_id = graph.compute(x=data
                                  , x_gene_id=gene_id)
    if not isinstance(data, np.ndarray):
        if data == None:
            print('Invalid Preprocessing Scheme is detected!!!')
    ad__ = ad.AnnData(pd.DataFrame(data=data, columns=gene_id))

    ad__.obs_names = ad_.obs_names
    ad__.obs = ad_.obs
    ad__.obsm = ad_.obsm
    ad__.write(extractor.save_path + 'trans_processed.h5',
               compression="gzip", compression_opts=9)