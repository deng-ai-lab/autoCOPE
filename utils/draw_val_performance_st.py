import numpy as np
import pandas as pd

def get_cos_pearson_spearman(path_dir, path_target, path_save):
    f = open(path_dir + path_target, 'r')
    reward_top1 = []
    reward_top5 = []
    reward_top20 = []
    cos_top1 = []
    pearson_top1 = []
    spearman_top1 = []
    cos_top5 = []
    pearson_top5 = []
    spearman_top5 = []
    cos_top20 = []
    pearson_top20 = []
    spearman_top20 = []
    for line in f:
        if ('arch_epoch' in line) and ('val' in line):
            print(line)
            line = line.rstrip('\n').split(' ')
            if 'top1_cos_score_val' in line:
                cos_top1.append(float(line[line.index('top1_cos_score_val')+1].split(';')[0]))
            if 'top1_pear_score_val' in line:
                pearson_top1.append(float(line[line.index('top1_pear_score_val')+1]))
            if 'top1_spearman_score_val' in line:
                spearman_top1.append(float(line[line.index('top1_spearman_score_val') + 1]))
            if 'top5_avg_cos_score_val' in line:
                cos_top5.append(float(line[line.index('top5_avg_cos_score_val')+1].split(';')[0]))
            if 'top5_avg_pear_score_val' in line:
                pearson_top5.append(float(line[line.index('top5_avg_pear_score_val')+1]))
            if 'top5_avg_spearman_score_val' in line:
                spearman_top5.append(float(line[line.index('top5_avg_spearman_score_val') + 1]))
            if 'top20_avg_cos_score_val' in line:
                cos_top20.append(float(line[line.index('top20_avg_cos_score_val')+1].split(';')[0]))
            if 'top20_avg_pear_score_val' in line:
                pearson_top20.append(float(line[line.index('top20_avg_pear_score_val')+1]))
            if 'top20_avg_spearman_score_val' in line:
                spearman_top20.append(float(line[line.index('top20_avg_spearman_score_val') + 1]))
        if ('arch_epoch' in line) and ('top1_reward' in line):
            print(line)
            line = line.rstrip('\n').split(' ')
            reward_top1.append(float(line[line.index('top1_reward')+1].split(';')[0]))
            reward_top5.append(float(line[line.index('top5_avg_reward') + 1].split(';')[0]))
            reward_top20.append(float(line[line.index('top20_avg_reward') + 1].split(';')[0]))
    result = np.concatenate([np.array(cos_top1).reshape(-1, 1)
                           , np.array(pearson_top1).reshape(-1, 1)
                           , np.array(spearman_top1).reshape(-1, 1)
                           , np.array(cos_top5).reshape(-1, 1)
                           , np.array(pearson_top5).reshape(-1, 1)
                           , np.array(spearman_top5).reshape(-1, 1)
                           , np.array(cos_top20).reshape(-1, 1)
                           , np.array(pearson_top20).reshape(-1, 1)
                           , np.array(spearman_top20).reshape(-1, 1)
                           , np.array(reward_top1).reshape(-1, 1)
                           , np.array(reward_top5).reshape(-1, 1)
                           , np.array(reward_top20).reshape(-1, 1)], axis = 1)
    #result = result.reshape(3, 50)
    #result = np.concatenate([result[:, 10:], result[:, :10]], axis = 1)
    columns = ['top1_cos_score_val'] + ['top1_pear_score_val'] + ['top1_spearman_score_val']\
            + ['top5_avg_cos_score_val']  + ['top5_avg_pear_score_val']  + ['top5_avg_spearman_score_val']\
            + ['top20_avg_cos_score_val']  + ['top20_avg_pear_score_val']  + ['top20_avg_spearman_score_val']\
            + ['top1_reward']+ ['top5_avg_reward']+ ['top20_avg_reward']
    result = pd.DataFrame(result, columns = columns)
    result.to_csv(path_dir + path_save)

path_dir = '/home/hym2/projects_dir/scPSAD_v11/ppo_search_20220301-205040/'
path_target = 'log.txt'
path_save = 'cos_pearson_spearman.csv'
get_cos_pearson_spearman(path_dir, path_target, path_save)