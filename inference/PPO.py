from architecture.autoCOPE import AUTOCOPE
from reward.Rewarder_cell_discrete import Rewarder as Rewarder_cell_discrete
from reward.Rewarder_cell_discrete import get_score as get_score_cell_discrete
from reward.Rewarder_cell_continuous import Rewarder as Rewarder_cell_continuous
from reward.Rewarder_cell_continuous import get_score as get_score_cell_continuous
from reward.Rewarder_gene_continuous import Rewarder as Rewarder_gene_continuous
from reward.Rewarder_gene_continuous import get_score as get_score_gene_continuous
import numpy as np
import torch
import torch.optim as optim
import logging
import os
from multiprocessing import Process, Queue
import multiprocessing
from architecture.preprocessing_scheme import PreprocessingScheme
multiprocessing.set_start_method('spawn', force=True)

#Setting 2 GPUS available for me.
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

def consume(rewarder, results_queue):
    if rewarder.args.task == 'cell_continuous':
        get_score_cell_continuous(rewarder)
    elif rewarder.args.task == 'cell_continuous':
        get_score_cell_discrete(rewarder)
    elif rewarder.args.task == 'gene_continuous':
        get_score_gene_continuous(rewarder)
    results_queue.put(rewarder)

class PPO_(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.ppo_epochs = args.ppo_epochs

        self.controller = AUTOCOPE(args, device=device).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight

        self.clip_epsilon = 0.2

    def multi_solve_environment(self):
        rewarders_top20 = []

        for arch_epoch in range(self.arch_epochs):
            results_queue = Queue()
            processes = []

            for episode in range(self.episodes):
                actions_p, actions_log_p, actions_index, ps_embedding, len_std_per_modality = self.controller.sample()
                actions_p = actions_p.cpu().numpy().tolist()
                actions_log_p = actions_log_p.cpu().numpy().tolist()
                actions_index = actions_index.cpu().numpy().tolist()

                if self.args.task == 'cell_continuous':
                    rewarder = Rewarder_cell_continuous(actions_p, actions_log_p, actions_index, len_std_per_modality, self.args)
                elif self.args.task == 'cell_discrete':
                    rewarder = Rewarder_cell_discrete(actions_p, actions_log_p, actions_index, len_std_per_modality, self.args)
                elif self.args.task == 'gene_continuous':
                    rewarder = Rewarder_gene_continuous(actions_p, actions_log_p, actions_index, len_std_per_modality,
                                                      self.args)
                else:
                    print('Param task wrong!!!!')

                #get_score_cell_discrete(rewarder)
                #results_queue.put(rewarder)
                process = Process(target=consume, args=(rewarder, results_queue))
                process.start()
                processes.append(process)
            
            for process in processes:
                process.join()

            rewarders = []
            for episode in range(self.episodes):
                rewarder = results_queue.get()
                rewarder.actions_p = torch.Tensor(rewarder.actions_p).to(self.device)
                rewarder.actions_index = torch.LongTensor(rewarder.actions_index).to(self.device)
                rewarders.append(rewarder)
            save_rewards = []
            for episode, rewarder in enumerate(rewarders):
                if self.args.task == 'cell_continuous':
                    logging.info(
                        'valid mse {:.4f}; valid r2 {:.4f}; valid cos {:.4f}; valid pear {:.4f}; valid spearman {:.4f}; valid reward {:.4f}'.format(
                            rewarder.mse, rewarder.r2, rewarder.cos, rewarder.spear, rewarder.spearman,
                            rewarder.reward))
                elif self.args.task == 'cell_discrete':
                    logging.info(
                        'valid median f1 {:.4f}; valid reward {:.4f}'.format(rewarder.f1, rewarder.reward))
                elif self.args.task == 'gene_continuous':
                    logging.info(
                        'valid mse {:.4f}; valid r2 {:.4f}; valid cos {:.4f}; valid pear {:.4f}; valid spearman {:.4f}; valid reward {:.4f}'.format(
                            rewarder.mse_val, rewarder.r2_val, rewarder.cos_score_val, rewarder.pear_score_val, rewarder.spearman_score_val,
                            rewarder.reward))
                logging.info('max length: ' + str(rewarder.args.graph_max_size) + '; current length: ' + str(
                    rewarder.len_std[0]) + ': length reward: ' + str(rewarder.len_score))
                save_rewards.append(rewarder.reward)
            mean_reward = np.mean(save_rewards)
            if self.baseline == None:
                self.baseline = mean_reward
            else:
                self.baseline = self.baseline * self.baseline_weight + mean_reward * (1 - self.baseline_weight)

            # sort worker retain top20
            rewarders_total = rewarders_top20 + rewarders
            rewarders_total.sort(key=lambda rewarder: rewarder.reward, reverse=True)
            rewarders_top20 = rewarders_total[:20]

            top1_reward = rewarders_top20[0].reward
            top5_avg_reward = np.mean([worker.reward for worker in rewarders_top20[:5]])
            top20_avg_reward = np.mean([worker.reward for worker in rewarders_top20])
            logging.info(
                'arch_epoch {:0>3d} top1_reward {:.4f} top5_avg_reward {:.4f} top20_avg_reward {:.4f} baseline {:.4f} '.format(
                    arch_epoch, top1_reward, top5_avg_reward, top20_avg_reward, self.baseline))

            top1_ari = rewarders_top20[0].reward
            top5_avg_ari = np.mean([worker.reward for worker in rewarders_top20[:5]])
            top20_avg_ari = np.mean([worker.reward for worker in rewarders_top20])
            logging.info('arch_epoch {:0>3d} top1_reward {:.4f} top5_avg_reward {:.4f} top20_avg_reward {:.4f}'.format(
                arch_epoch, top1_ari, top5_avg_ari, top20_avg_ari))

            for i in range(20):
                graph = PreprocessingScheme(rewarders_top20[i].args, rewarders_top20[i].actions_index, rewarders_top20[i].len_std)
                graph.post_order()
                logging.info(graph.operation_path)

            for ppo_epoch in range(self.ppo_epochs):
                loss = 0

                for rewarder in rewarders:
                    actions_p, actions_log_p = self.controller.get_p(rewarder.actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, rewarder, self.baseline)

                loss /= len(rewarders)
                logging.info('ppo_epoch {:0>3d} loss {:.4f} '.format(ppo_epoch, loss))

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()

            # save LSTM weight
            try:
                torch.save(obj=self.controller.state_dict(), f=self.args.top_5_save_dir + "epoch_" + str(arch_epoch) + "_" + "LSTM.pth")
            except:
                pass
            # Save the top 5 computational graph for further check.
            if arch_epoch % self.args.check_point_epoch == (self.args.check_point_epoch - 1):
                path = self.args.top_5_save_dir
                for i in range(5):
                    graph = PreprocessingScheme(rewarders_top20[i].args, rewarders_top20[i].actions_index, rewarders_top20[i].len_std)
                    np.save(path + "epoch_" + str(arch_epoch) + "_top_" + str(i) + "_graph", graph.doubly_linked_list)


    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance

    def cal_loss(self, actions_p, actions_log_p, rewarder, baseline):
        actions_importance = actions_p / rewarder.actions_p
        clipped_actions_importance = self.clip(actions_importance)
        if rewarder.reward != rewarder.args.nan_penalty:
            reward = rewarder.reward - baseline
        else:
            reward = rewarder.args.nan_penalty
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight


        print('actions_importance', actions_importance)
        print('clipped_actions_importance', clipped_actions_importance)
        print('policy_loss', policy_loss)
        print('entropy_bonus', entropy_bonus)


        return policy_loss + entropy_bonus