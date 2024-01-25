import numpy as np
import pandas as pd

def get_cos_pearson_spearman(path_dir, path_target, path_save):
    f = open(path_dir + path_target, 'r')
    reward_top1 = []
    reward_top5 = []
    reward_top20 = []
    for line in f:
        if ('arch_epoch' in line) and ('top1_reward' in line):
            print(line)
            line = line.rstrip('\n').split(' ')
            reward_top1.append(float(line[line.index('top1_reward')+1].split(';')[0]))
            reward_top5.append(float(line[line.index('top5_avg_reward') + 1].split(';')[0]))
            reward_top20.append(float(line[line.index('top20_avg_reward') + 1].split(';')[0]))
    result = np.concatenate([np.array(reward_top1).reshape(-1, 1)
                           , np.array(reward_top5).reshape(-1, 1)
                           , np.array(reward_top20).reshape(-1, 1)], axis = 1)
    #result = result.reshape(3, 50)
    #result = np.concatenate([result[:, 10:], result[:, :10]], axis = 1)
    columns = ['top1_reward']+ ['top5_avg_reward']+ ['top20_avg_reward']
    result = pd.DataFrame(result, columns = columns)
    result.to_csv(path_dir + path_save)

path_dir = '/home/hym2/projects_dir/scPSAD_v11/pg_search_20220223-151231/'
path_target = 'log.txt'
path_save = 'reward.csv'
get_cos_pearson_spearman(path_dir, path_target, path_save)