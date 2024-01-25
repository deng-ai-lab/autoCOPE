import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_acc(path_dir, path_target, path_save):
    f = open(path_dir + path_target, 'r')
    ari = []
    silhouette = []
    for line in f:
        if 'valid' in line:
            print(line)
            line = line.rstrip('\n').split(' ')
            ari.append(float(line[line.index('ari')+1].split(';')[0]))
            silhouette.append(float(line[line.index('silhouette_score')+1]))
    result = np.concatenate([np.array(ari).reshape(1, -1)
                                        , np.array(silhouette).reshape(1, -1)], axis = 0)
    result = result.reshape(2, 30)
    result = np.concatenate([result[:, 10:], result[:, :10]], axis = 1)
    columns = ['count_' + str(i) for i in range(10)]\
              + ['nmeth_' + str(i) for i in range(10)]\
              + ['DCA_' + str(i) for i in range(10)]
    result = pd.DataFrame(result, columns = columns)
    result.to_csv(path_dir + path_save)

path_dir = '/home/hym2/project_dir/scPSAD_v10/analysis/evaluate_dataset_20211231-163837/'
path_target = 'log.txt'
path_save = 'ari_silhouette.csv'
get_acc(path_dir, path_target, path_save)