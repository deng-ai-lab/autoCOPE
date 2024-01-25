import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import special
from sklearn.decomposition import NMF
from sklearn.decomposition import FactorAnalysis
#import scscope as DeepImpute
#import scanorama
from sklearn.decomposition import PCA
import random
#import tensorflow as tf
import umap


class unit_lib(object):
    def __init__(self, args):
        # 1. Define all possible nodes in a computational graph seperately.
        self.node_feature_selection = ['gene_filter_top_hvgs_2000', 'gene_filter_top_hvgs_2200', 'gene_filter_top_hvgs_2400'
                                     , 'gene_filter_top_hvgs_2600', 'gene_filter_top_hvgs_2800', 'gene_filter_top_hvgs_3000'
                                     , 'gene_filter_top_hvgs_3200', 'gene_filter_top_hvgs_3400', 'gene_filter_top_hvgs_3600']
        self.node_scaling = ['log', 'sqrt', 'alf', 'erf', 'gd', 'tanh', 'standardization']
        self.node_normalization = ['CPM', 'TMM', 'MRN', 'RLE', 'UQN', 'Scran', 'TPM', 'Linnorm']
        self.node_stop = ['stop']

        self.node_multi_omics_integration = ['Combat', 'MNN', 'Scanorama', 'Harmony']#, 'Seurat_v3']

        self.node_dimensionality_reduction = ['PCA_5', 'PCA_10', 'PCA_15', 'PCA_20', 'PCA_25', 'PCA_30', 'PCA_35', 'PCA_40', 'PCA_45'
                                            , 'NMF_5', 'NMF_10', 'NMF_15', 'NMF_20', 'NMF_25', 'NMF_30', 'NMF_35', 'NMF_40', 'NMF_45'
                                            , 'FA_5', 'FA_10', 'FA_15', 'FA_20', 'FA_25', 'FA_30', 'FA_35', 'FA_40', 'FA_45'
                                            , 'DiffusionMap_5', 'DiffusionMap_10', 'DiffusionMap_15', 'DiffusionMap_20', 'DiffusionMap_25', 'DiffusionMap_30'
                                            , 'DiffusionMap_35', 'DiffusionMap_40', 'DiffusionMap_45'
                                            , 'UMAP_5', 'UMAP_10', 'UMAP_15', 'UMAP_20', 'UMAP_25', 'UMAP_30', 'UMAP_35', 'UMAP_40', 'UMAP_45'
                                            , 'scScope_5', 'scScope_10', 'scScope_15', 'scScope_20', 'scScope_25', 'scScope_30', 'scScope_35', 'scScope_40', 'scScope_45']

        # 2. Define node sets in the output of scPSAD.
        self.gene_min_size = 1500
        if args.Hdf5Path_target != '':
            num_gene = sc.read(args.Hdf5Path_target).X.shape[1]
        else:
            num_gene = sc.read(args.Hdf5Path_ref).X.shape[1]
        if num_gene < self.gene_min_size or num_gene == self.gene_min_size:
            self.nodes_std = self.node_scaling + self.node_normalization + self.node_stop
        else:
            self.nodes_std = self.node_feature_selection + self.node_scaling + self.node_normalization + self.node_stop
        self.nodes_moi = self.node_multi_omics_integration
        self.nodes_dr = self.node_dimensionality_reduction

        # 3. Define the interpretation between parameters from network output and operation input.
        self.parameter_dim = {'2000': 2000, '2200': 2200, '2400': 2400, '2600': 2600, '2800': 2800, '3000': 3000, '3200': 3200, '3400': 3400, '3600': 3600
                            , '5': 5, '10': 10, '15': 15, '20': 20, '25': 25, '30': 30, '35': 35, '40': 40, '45': 45}

        # 4. Define essential functions from r language.
        # 4.1 Multi-omics integration for seurat.
        from rpy2.robjects.packages import importr

        import rpy2.robjects as robj
        self.robj = robj
        self.list_r = robj.r('function(x, y) list(x, y)')
        self.gene_id = robj.r("function(start, end) paste0('gene_', start:end)")
        self.cell_id = robj.r("function(start, end) paste0('cell_', start:end)")

        # 4.2 Normalization Node.
        # 4.2.1 scran Operation.
        self.scran_package = importr('scran')
        self.EBSeq_package = importr('EBSeq')
        self.sce = robj.r('function(x) computeSumFactors(SingleCellExperiment(assays = list(counts = x)), positive = FALSE)')
        self.sce_sum = robj.r('function(x) GetNormalizedMat(assay(x, "counts"),sizeFactors(x))')
        # 4.2.2 linnorm Operation.
        self.linnorm_package = importr('Linnorm')
        self.linnorm = robj.r('function(x) Linnorm(x, minNonZeroPortion = 0.0)')
        # 4.2.3 SCTransform Operation.
        self.sct = robj.r('function(x) SCTransform(x, residual.features = rownames(x = x))')
        self.sct_extraction = robj.r('function(x) as.matrix(x@assays$SCT@data)')
        # 4.2.4 TPM Operation.
        self.sce_package = importr('SingleCellExperiment')
        self.CreateSCEObject = robj.r('function(x) SingleCellExperiment(assays = list(counts = x))')
        self.tpm_func = robj.r('function(x) calculateTPM(x, lengths = 5e4)')
        # 4.2.5 UQN Operation.
        self.uqn_func = robj.r('''function(expr_mat) {
                                                 UQ <- function(x) {quantile(x[x > 0], 0.75)}
                                                uq <- unlist(apply(expr_mat, 2, UQ))
                                                norm_factor <- uq/median(uq)
                                                return(t(t(expr_mat)/norm_factor))}''')
        # 4.2.6 RLE Operation
        self.rle_func = robj.r('''function(expr_mat) {
                                                 geomeans <- exp(rowMeans(log(expr_mat)))
                                                 SF <- function(cnts) {median((cnts/geomeans)[(is.finite(geomeans) & geomeans > 0)])}
                                                norm_factor <- apply(expr_mat, 2, SF)
                                                return(t(t(expr_mat)/norm_factor))}''')
        # 4.2.7 TMM Operation.
        self.edger_package = importr('edgeR')
        self.tmm_func = robj.r('''function(expr_mat) {
                                                        norm_factor <- calcNormFactors(expr_mat,method = "TMM")
                                                        return(t(t(expr_mat)/norm_factor))}''')
        # 4.2.8 MRN Operation.
        self.mrn_func = robj.r('''function(expr_mat) {
                                                        norm_factor <- MedianNorm(expr_mat)
                                                        return(t(t(expr_mat)/norm_factor))}''')

        # 5. Define essential limitation for computational scalability.
        self.max_genes_nl = 3500
        self.max_genes_bc = 2000
        self.seed = args.seed_test


    def get_operation(self, opt_name, x, x_gene_id = None, y = None, y_gene_id = None):
        if (opt_name in self.node_dimensionality_reduction) or (opt_name in self.node_feature_selection):
            opt_name, dim_parameter = opt_name.rsplit('_', 1)
            dim_parameter = self.parameter_dim[dim_parameter]
        else:
            dim_parameter = None

        return getattr(self, opt_name)(x, x_gene_id, y, y_gene_id, dim_parameter)


    def take_intersection(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Take the intersection of the two array.
        if y.any() == None and y_gene_id.any() == None:
            adata_concat = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        else:
            x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
            y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
            var_names = x_data.var_names.intersection(y_data.var_names)
            x_data = x_data[:, var_names]
            y_data = y_data[:, var_names]
            adata_concat = x_data.concatenate(y_data, batch_categories=['sample_1', 'sample_2'])

        return adata_concat


    def create_matrix_r(self, x, x_gene_id, x_cell_start_id):
        data = x.transpose()  # row for genes and col for cells.
        nr, nc = data.shape
        data_vec = self.robj.FloatVector(data.transpose().reshape((data.size)))
        cell_id = self.cell_id(x_cell_start_id, x_cell_start_id + x.shape[0] - 1)
        gene_id = self.gene_id(0, 0 + x.shape[1] - 1)
        data_r = self.robj.r.matrix(data_vec, nrow=nr, ncol=nc, dimnames=self.list_r(gene_id, cell_id))

        return data_r


    # def create_seurat_object(self, x, x_gene_id, x_cell_start_id):
    #     # 1. Create R object.
    #     data_r = self.create_matrix_r(x, x_gene_id, x_cell_start_id)
    #
    #     # 2. Return seurat object.
    #     obj = self.CreateSeuratObject(data_r)
    #     obj = self.seurat_norm(obj)
    #     obj = self.set_variable_features(obj)
    #     return obj


    def PCA(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        try:
            pca = PCA(n_components=dim_parameter, svd_solver='full', random_state = self.seed)
            x_data = pca.fit_transform(x)
        except:
            return None, None
        x_gene_id = ['pca_' + str(i) for i in range(dim_parameter)]

        return x_data, x_gene_id


    def NMF(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.
        # 1. Take the abs of the input data x to satisfy the requierment that input should be nonnegative.
        print(x.shape)
        x = np.abs(x)

        # 2. Take the NMF operation.
        x_adata = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        try:
            model = NMF(n_components=dim_parameter, random_state=self.seed, max_iter=300)
            x_data = model.fit_transform(x_adata.X)
        except:
            return None, None
        x_gene_id = ['nmf_' + str(i) for i in range(dim_parameter)]

        return x_data, x_gene_id


    def FA(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        x_adata = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        try:
            model = FactorAnalysis(n_components=dim_parameter, random_state=self.seed, max_iter=300)
            x_data = model.fit_transform(x_adata.X)
        except:
            return None, None
        x_gene_id = ['fa_' + str(i) for i in range(dim_parameter)]

        return x_data, x_gene_id


    # def DiffusionMap(self, x, x_gene_id, y, y_gene_id, dim_parameter):
    #     # Input data structure: ndarrray.
    #
    #     x_adata = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
    #     try:
    #         sc.pp.neighbors(x_adata, use_rep='X', random_state = self.seed)
    #         sc.tl.diffmap(x_adata, n_comps=dim_parameter, random_state = self.seed)
    #     except:
    #         return None, None
    #     x_data = x_adata.obsm['X_diffmap']
    #     x_gene_id = ['diffmap_' + str(i) for i in range(dim_parameter)]
    #
    #     return x_data, x_gene_id


    # def scScope(self, x, x_gene_id, y, y_gene_id, dim_parameter):
    #     # Input data structure: ndarrray.
    #     np.random.seed(self.seed)
    #     random.seed(self.seed)
    #     tf.random.set_seed(self.seed)
    #
    #     x = np.abs(x)
    #
    #     x_adata = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
    #     try:
    #         model = DeepImpute.train(x_adata.X, latent_code_dim = dim_parameter, epoch_per_check = 50)
    #         x_data, imputed_val, _ = DeepImpute.predict(x_adata.X, model)
    #     except:
    #         return None, None
    #     x_data = x_data[:, dim_parameter:]    # Take the vector from the last recurrent component as the latent embedding.
    #     x_gene_id = ['scScope_' + str(i) for i in range(dim_parameter)]
    #
    #     return x_data, x_gene_id

    def UMAP(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.
        print(x.shape)
        data, gene_id = self.PCA(x, x_gene_id, y, y_gene_id, 50)

        x_data = umap.UMAP(n_components = dim_parameter, random_state = self.seed).fit_transform(data)
        x_gene_id = ['umap_' + str(i) for i in range(dim_parameter)]

        return x_data, x_gene_id


    def Combat(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        # 0. Take intersection of two datasets.
        x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
        var_names = x_data.var_names.intersection(y_data.var_names)
        x_data = x_data[:, var_names]
        y_data = y_data[:, var_names]
        x, x_gene_id, y, y_gene_id = x_data.X, x_data.var_names, y_data.X, y_data.var_names


        # 1. Check for the scalability of the dataset.
        #if x.shape[1] > self.max_genes_bc:
        #    x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes)
        #if y.shape[1] > self.max_genes_bc:
        #    y, y_gene_id = self.gene_filter_top_hvgs(y, y_gene_id, None, None, self.max_genes)

        # 2. Execute the multi-omics integration operation.
        adata_concat = self.take_intersection(x, x_gene_id, y, y_gene_id, dim_parameter)
        try:
            sc.pp.combat(adata_concat, key='batch')
        except:
            return None, None
        data = adata_concat.X
        gene_id = adata_concat.var_names.values.tolist()

        return data, gene_id


    def MNN(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        # 0. Take intersection of two datasets.
        x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
        var_names = x_data.var_names.intersection(y_data.var_names)
        x_data = x_data[:, var_names]
        y_data = y_data[:, var_names]
        x, x_gene_id, y, y_gene_id = x_data.X, x_data.var_names, y_data.X, y_data.var_names


        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_bc:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_bc)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None
        if y.shape[1] > self.max_genes_bc:
            y, y_gene_id = self.gene_filter_top_hvgs(y, y_gene_id, None, None, self.max_genes_bc)
            if not isinstance(y, np.ndarray):
                if y == None:
                    return None, None

        # 2. Load individual dataset.
        x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
        var_names = x_data.var_names.intersection(y_data.var_names)
        x_data = x_data[:, var_names]
        y_data = y_data[:, var_names]

        # 3. Multi-omics Integration.
        try:
            adatas_cor = sc.external.pp.mnn_correct(x_data, y_data, batch_key='batch')
        except:
            return None, None

        # 4. Collect the integrated counts.
        data = adatas_cor[0].X
        gene_id = var_names.values.tolist()

        return data, gene_id


    # def Scanorama(self, x, x_gene_id, y, y_gene_id, dim_parameter):
    #
    #     # 0. Take intersection of two datasets.
    #     x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
    #     y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
    #     var_names = x_data.var_names.intersection(y_data.var_names)
    #     x_data = x_data[:, var_names]
    #     y_data = y_data[:, var_names]
    #     x, x_gene_id, y, y_gene_id = x_data.X, x_data.var_names, y_data.X, y_data.var_names
    #
    #     # 1. Check for the scalability of the dataset.
    #     if x.shape[1] > self.max_genes_bc:
    #         x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_bc)
    #         if not isinstance(x, np.ndarray):
    #             if x == None:
    #                 return None, None
    #     if y.shape[1] > self.max_genes_bc:
    #         y, y_gene_id = self.gene_filter_top_hvgs(y, y_gene_id, None, None, self.max_genes_bc)
    #         if not isinstance(y, np.ndarray):
    #             if y == None:
    #                 return None, None
    #
    #     # 2. Load individual dataset.
    #     x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
    #     y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
    #     var_names = x_data.var_names.intersection(y_data.var_names)
    #     x_data = x_data[:, var_names]
    #     y_data = y_data[:, var_names]
    #     adatas = [x_data, y_data]
    #
    #     # 3. Multi-omics Integration.
    #     try:
    #         adatas_cor = scanorama.correct_scanpy(adatas, return_dimred=True, dimred = len(var_names.values.tolist()), seed = self.seed)
    #     except:
    #         return None, None
    #
    #     # 4. Collect the integrated counts.
    #     data = np.concatenate((adatas_cor[0].obsm['X_scanorama'], adatas_cor[1].obsm['X_scanorama']), axis=0)
    #     gene_id = var_names.values.tolist()
    #
    #     return data, gene_id


    def Harmony(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        # 0. Take intersection of two datasets.
        x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
        var_names = x_data.var_names.intersection(y_data.var_names)
        x_data = x_data[:, var_names]
        y_data = y_data[:, var_names]

        x, x_gene_id, y, y_gene_id = x_data.X, x_data.var_names, y_data.X, y_data.var_names

        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_bc:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_bc)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None
        if y.shape[1] > self.max_genes_bc:
            y, y_gene_id = self.gene_filter_top_hvgs(y, y_gene_id, None, None, self.max_genes_bc)
            if not isinstance(y, np.ndarray):
                if y == None:
                    return None, None

        # 2. Execute the multi-omics integration operation.
        adata_concat = self.take_intersection(x, x_gene_id, y, y_gene_id, dim_parameter)
        adata_concat.obsm['X'] = adata_concat.X + 1e-10     # Add a small number to avoid exceptions in harmony.
        try:
            sc.external.pp.harmony_integrate(adata_concat, key='batch', basis='X', random_state = self.seed)
        except:
            return None, None

        data = adata_concat.obsm['X_pca_harmony']
        gene_id = adata_concat.var_names.values.tolist()

        return data, gene_id

    #def Seurat_v3(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        # 0. Take intersection of two datasets.
    #    x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
    #    y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
    #    var_names = x_data.var_names.intersection(y_data.var_names)
    #    x_data = x_data[:, var_names]
    #    y_data = y_data[:, var_names]
    #    x, x_gene_id, y, y_gene_id = x_data.X, x_data.var_names, y_data.X, y_data.var_names

        # 1. Check for the scalability of the dataset.
        #if x.shape[1] > self.max_genes_bc:
        #    x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_bc)
        #    if not isinstance(x, np.ndarray):
        #        if x == None:
        #            return None, None
        #if y.shape[1] > self.max_genes_bc:
        #    y, y_gene_id = self.gene_filter_top_hvgs(y, y_gene_id, None, None, self.max_genes_bc)
        #    if not isinstance(y, np.ndarray):
        #        if y == None:
        #            return None, None

        # 2. Take intersection of two datasets.
        #x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        #y_data = ad.AnnData(pd.DataFrame(data=y, columns=y_gene_id))
        #var_names = x_data.var_names.intersection(y_data.var_names)
        #x_data = x_data[:, var_names]
        #y_data = y_data[:, var_names]
        #x, x_gene_id, y, y_gene_id = x_data.X, x_data.var_names, y_data.X, y_data.var_names

        # 3. Execute the multi-omics integration operation.
        # 3.1 Create Seurat Object.
    #    x_cell_start_id = 0
    #    y_cell_start_id = x.shape[0]
    #    x_seurat = self.create_seurat_object(x, x_gene_id, x_cell_start_id)
    #    y_seurat = self.create_seurat_object(y, y_gene_id, y_cell_start_id)

        # 3.2 Integrate Dataset.
    #    try:
    #        integrated = self.integrate_seurat(self.list_r(x_seurat, y_seurat))
    #    except:
    #        return None, None

        # 3.3 Extract the result of integration.
    #    if integrated == self.robj.rinterface.NULL:
    #        print('Seurat Time Out Error!!!!!!!!!!!!!!!!!!')
    #        return None, None
    #    print('Seurat Time ok!!!!!!!!!!!!!!!!!!')
    #    concat_data = np.array(list(self.list_extraction(integrated))).reshape(x.shape[1], x.shape[0] + y.shape[0]).transpose()
    #    concat_gene_id = x_gene_id.values.tolist()

    #    return concat_data, concat_gene_id

    def gene_filter_top_hvgs(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.
        x_data = ad.AnnData(pd.DataFrame(data=x, columns=x_gene_id))
        try:
            sc.pp.highly_variable_genes(x_data, n_top_genes=dim_parameter
                                        , flavor='cell_ranger')
        except:
            return None, None
        x_data = x_data[:, x_data.var.highly_variable.values]
        x_gene_id = x_data.var_names.values.tolist()

        return x_data.X, x_gene_id

    def log(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        return np.sign(x) * np.log2(np.abs(x) + np.finfo(np.float32).eps)

    def sqrt(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        return np.sign(x) * np.sqrt(np.abs(x) + np.finfo(np.float32).eps)

    def alf(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.
        temp = np.sqrt(1 + x * x)

        return x / temp

    def erf(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        return special.erf(x)


    def gd(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        return 2 * np.arctan(np.tanh(x / 2))

    def tanh(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.

        return np.tanh(x)

    def standardization(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.
        x_data = ad.AnnData(pd.DataFrame(data=x))
        try:
            sc.pp.scale(x_data, max_value=10)
        except:
            return None

        return x_data.X

    def CPM(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # Input data structure: ndarrray.
        x_data = ad.AnnData(pd.DataFrame(data=x))
        try:
            sc.pp.normalize_total(x_data, target_sum=1e6)
        except:
            return None, None

        return x_data.X, x_gene_id

    def TMM(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Create r object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)

        # 3. Execute the normalization operation.
        try:
            x = self.tmm_func(x)
            x = np.array(x).transpose()
        except:
            return None, None

        return x, x_gene_id

    def MRN(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Check positive property
        x = np.abs(x)

        # 3. Create r object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)

        # 4. Execute the normalization operation.
        try:
            x = self.mrn_func(x)
            x = np.array(x).transpose()
        except:
            return None, None

        return x, x_gene_id

    def RLE(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Check positive property
        x = np.abs(x)

        # 3. Create r object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)

        # 4. Execute the normalization operation.
        try:
            x = self.rle_func(x)
            x = np.array(x).transpose()
        except:
            return None, None
        return x, x_gene_id

    def UQN(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Check positive property
        x = np.abs(x)

        # 3. Create r object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)

        # 4. Execute the normalization operation.
        try:
            x = self.uqn_func(x)
            x = np.array(x).transpose()
        except:
            return None, None

        return x, x_gene_id

    def Scran(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Check positive property
        x = np.abs(x)

        # 3. Create r object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)

        # 4. Execute the normalization operation.
        try:
            x = self.sce(x)
            x = self.sce_sum(x)
            x = np.array(x).transpose()
        except:
            return None, None

        return x, x_gene_id

    def TPM(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Check positive property
        x = np.abs(x)

        # 3. Create R object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)
        x = self.CreateSCEObject(x)

        # 4. Execute the normalization operation.
        try:
            x = self.tpm_func(x)
            x = np.array(x).transpose()
        except:
            return None, None

        return x, x_gene_id

    def Linnorm(self, x, x_gene_id, y, y_gene_id, dim_parameter):
        # 1. Check for the scalability of the dataset.
        if x.shape[1] > self.max_genes_nl:
            x, x_gene_id = self.gene_filter_top_hvgs(x, x_gene_id, None, None, self.max_genes_nl)
            if not isinstance(x, np.ndarray):
                if x == None:
                    return None, None

        # 2. Check positive property
        x = np.abs(x)

        # 3. Create r object.
        x_cell_start_id = 0
        x = self.create_matrix_r(x, x_gene_id, x_cell_start_id)

        # 4. Execute the normalization operation.
        try:
            x = self.linnorm(x)
        except:
            return None, None
        x = np.array(x).transpose()

        return x, x_gene_id