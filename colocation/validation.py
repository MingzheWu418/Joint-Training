from ddqn import DQN, EpisodeHistory
from networks import *
import numpy as np
import torch
from tqdm import tqdm
# from tensorflow.summary import create_file_writer, scalar
# import tensorflow as tf
import datetime
from read_data import write_result
import matplotlib.pyplot as plt

class test_summary:

    def __init__(self, alg, graph_generator=None, action_type='swap', q_net='mlp', forbid_revisit=False):
        if isinstance(alg, DQN):
            self.alg = alg.model
        else:
            self.alg = alg

        self.graph_generator = graph_generator
        self.n = graph_generator.n
        self.action_type = action_type
        self.episodes = []
        self.S = []
        self.max_gain = []
        self.max_gain_budget = []
        self.max_gain_ratio = []
        self.q_net = q_net
        self.forbid_revisit = forbid_revisit
        self.state_eval = []
        self.bg = []
        self.final_value = 0
        self.bg_list = []
        self.batch_size = 100
        self.acc = 0
                        # 40
    def run_test(self, problem=None, init_trial=10, trial_num=1, batch_size=100, action_dropout=1.0, gnn_step=3, episode_len=50, explore_prob=0.1, Temperature=1.0, action_reserve_ratio=1.0, cuda_flag=True, save_log=False, k=10, m=4):
        
        # if save_log:
        #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
        #     test_log_dir = './log/testing_read_model/' + current_time + '/'

        #     # Initialize tensorboard
        #     test_summary_writer = create_file_writer(test_log_dir)

        self.batch_size = batch_size
        self.episode_len = episode_len
        self.trial_num = trial_num
        self.batch_size = batch_size
        if problem is None:
            batch_size *= trial_num
            bg = self.graph_generator.generate_graph(batch_size=self.batch_size, cuda_flag=cuda_flag)
            # print(bg)
            self.bg = self.graph_generator.generate_graph(x=bg.ndata['x'].repeat(trial_num, 1), batch_size=batch_size, cuda_flag=cuda_flag)
            gl = un_batch(self.bg)
        else:
            batch_size *= trial_num
            new_x = problem.ndata['x'].clone().detach()
            new_label = problem.ndata['label'].clone().detach()
            self.bg = self.graph_generator.generate_graph(x=new_x, batch_size=batch_size, cuda_flag=cuda_flag, label_input=new_label)
            gl = un_batch(self.bg)
            # print(self.bg)

        self.S = self.bg.kcut_value.clone()
        # print(self.S)

        self.last_actions = torch.zeros(batch_size, 2)
        self.second_last_actions = torch.zeros(batch_size, 2)
        self.third_last_actions = torch.zeros(batch_size, 2)

        ep = [EpisodeHistory(gl[i], max_episode_len=episode_len, action_type='swap') for i in range(batch_size)]
        self.end_of_episode = {}.fromkeys(range(batch_size), episode_len-1)
        loop_start_position = [0] * batch_size

        # TODO: how to implement forbid_revisit? The difficulty is when batch_size != 1
        forbidden_action = []
        # mean_Q = []

        for i in tqdm(range(episode_len)):
            
            batch_legal_actions = get_legal_actions(states=self.bg, action_type=self.action_type, action_dropout=action_dropout, m=self.graph_generator.m)
            forbid_action_mask = torch.zeros(batch_legal_actions.shape[0], 1)
            if self.bg.in_cuda:
                forbid_action_mask = forbid_action_mask.cuda()
            batch_legal_actions = batch_legal_actions * (1 - forbid_action_mask.int()).t().flatten().unsqueeze(1)

            prop_a_score, prop_actions = self.alg.forward_prop(self.bg, batch_legal_actions, top_ratio=action_reserve_ratio)
            # print(prop_actions)
            self.num_actions = prop_actions.shape[0] // batch_size

            self.action_mask = torch.tensor(range(0, self.num_actions * batch_size, self.num_actions))

            if self.bg.in_cuda:
                self.action_mask = self.action_mask.cuda()

            # Q_sa is the Q value estimation for each action
            """
            if i == 0:
                S_a_encoding, h1, h2, Q_sa = self.alg.forward(self.bg, prop_actions, action_type=self.action_type, gnn_step=gnn_step, save_figure=True)
            else:
                S_a_encoding, h1, h2, Q_sa = self.alg.forward(self.bg, prop_actions, action_type=self.action_type, gnn_step=gnn_step)
            """
            S_a_encoding, h1, h2, Q_sa = self.alg.forward(self.bg, prop_actions, action_type=self.action_type, gnn_step=gnn_step)
            # if torch.max(Q_sa) < 0:
            #     break
            #     # ~20-30 actions
            # print(Q_sa)

            # mean_Q.append(torch.mean(Q_sa).item())

            terminate_episode = (Q_sa.view(-1, self.num_actions).max(dim=1).values < 0.).nonzero().flatten().cpu().numpy()
            # print(terminate_episode)
            for idx in terminate_episode:
                if self.end_of_episode[idx] == episode_len - 1:
                    self.end_of_episode[idx] = i

            # My version, enables exploration
            top_ten_q_values, top_ten_indices = torch.topk(Q_sa.view(-1, self.num_actions), 10)
            best_actions_index = torch.multinomial(F.softmax(top_ten_q_values, dim=1), 1).item()
            best_actions = top_ten_indices[0][best_actions_index].view(-1)

            # Original version, disables exploration
            # best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, self.num_actions) / Temperature), 1).view(-1)
            
            chose_actions = torch.tensor(
                [x if torch.rand(1) > explore_prob else torch.randint(high=self.num_actions, size=(1,)).squeeze() for x in
                 best_actions])
            # print(chose_actions)
            if self.bg.in_cuda:
                chose_actions = chose_actions.cuda()
            chose_actions += self.action_mask
            if self.bg.in_cuda:
                chose_actions = chose_actions.cuda()
            actions = prop_actions[chose_actions]

            # Explore when in a loop
            # for g in range(batch_size):
            #     if (actions[g][0].item() == self.last_actions[g][0].item() and actions[g][1].item() == self.last_actions[g][1].item()) or (actions[g][0].item() == self.second_last_actions[g][0].item() and actions[g][1].item() == self.second_last_actions[g][1].item()) or (actions[g][0].item() == self.third_last_actions[g][0].item() and actions[g][1].item() == self.third_last_actions[g][1].item()):
            #         top_ten_q_values, top_ten_indices = torch.topk(Q_sa.view(-1, self.num_actions), 5)
            #         best_actions_index = torch.multinomial(F.softmax(top_ten_q_values, dim=1), 1).item()
            #         best_actions[g] = top_ten_indices[g][best_actions_index].view(-1)
            #         chose_actions[g] = best_actions[g]
            #         # print(chose_actions)
            #         chose_actions[g] += self.action_mask[g]
            #         actions[g] = prop_actions[chose_actions]
            # print(actions)

            self.third_last_actions = self.second_last_actions
            self.second_last_actions = self.last_actions
            self.last_actions = actions

            new_states, rewards = step_batch(states=self.bg, action=actions, action_type=self.action_type)

            self.bg_list.append(dc(new_states))
            R = [reward.item() for reward in rewards]
            # print(rewards)
            # print(R)
            self.bg = new_states
            # acc, _ = write_result(batch_size, k, m, new_states)
            # print(new_states)
            # print(best_actions[0]-self.action_mask[0])
            # print(actions)
            [ep[k].write(action=actions[k, :], action_idx=best_actions[k] - self.action_mask[k], reward=R[k]
                     , q_val=Q_sa.view(-1, self.num_actions)[k, :]
                     , actions=batch_legal_actions.view(-1, self.num_actions, 2)[k, :, :]
                     , state_enc=None
                     , sub_reward=None
                     , loop_start_position=loop_start_position[k]) for k in range(batch_size)]
            # print(batch_legal_actions.view(-1, self.num_actions, 2))
            # print(ep[0])

            # Tensorboard
            # if save_log:
            #     with test_summary_writer.as_default():
            #         scalar('Reward sequence', R[0], step=i)
            #         scalar('Remaining Edge', self.bg.kcut_value[0], step=i)
            #         scalar('Accuracy', acc, step=i)

            # if self.forbid_revisit:
            #     forbidden_action.append(chose_actions)

        self.episodes = ep

    def show_result(self, k, m):

        initial_S = []
        episode_end_S = []
        episode_gains = []
        episode_max_gains = []
        episode_gain_ratios = []

        for i in range(self.batch_size * self.trial_num):
            episode_max_gains.append(max(max(np.cumsum(self.episodes[i].reward_seq)), 0))
            episode_end_S.append( self.S[i].item() - max(episode_max_gains[-1], 0) )
        bs = self.batch_size
        x = [bs * np.argmin(episode_end_S[i::bs]) for i in range(bs)]
        select_indices = [x[i] + i for i in range(bs)]
        episode_max_gains = []
        episode_max_gain_steps = []
        episode_max_gain_ratios = []
        episode_end_value = []
        for i in select_indices:
            # print("-----")
            # print(self.S)
            # print(self.episodes[i].reward_seq)
            # print(self.episodes[i].action_seq)
            # print(self.episodes[i].enc_state_seq)
            initial_S.append(self.S[i].item())
            episode_gains.append(self.S[i].item() - sum(self.episodes[i].reward_seq[:self.end_of_episode[i]]))
            # episode_max_gains.append(self.S[i].item() - max(max(np.cumsum(self.episodes[i].reward_seq)), 0))
            episode_max_gains.append(self.S[i].item() - max(max(np.cumsum(self.episodes[i].reward_seq)), 0))
            # print(self.episodes[i].reward_seq)

            # Notice here we plus one to fit human intuition
            episode_max_gain_steps.append(np.argmax(np.cumsum(self.episodes[i].reward_seq)) + 1)
            episode_gain_ratios.append(episode_gains[-1] / self.S[i].item())
            episode_max_gain_ratios.append(max(episode_max_gains[-1], 0) / self.S[i].item())
            episode_end_value.append(episode_end_S[i])
        # print(episode_max_gain_steps)

        print('Avg value of initial S:', torch.mean(self.S).item())
        print('Avg episode end value:', np.mean(episode_gains))
        print('Avg episode best value:', np.mean(episode_max_gains))
        print('Avg episode step budget(Avg/Max/Min):', np.mean(episode_max_gain_steps), np.max(episode_max_gain_steps), np.min(episode_max_gain_steps))
        print('Avg percentage episode gain:', 1 - self.trial_num * sum(episode_gains) / sum(self.S).item())  #np.mean(episode_gain_ratios))
        print('Avg percentage max gain:', 1 - self.trial_num * sum(episode_max_gains) / sum(self.S).item())  #np.mean(episode_max_gain_ratios))
        print('Percentage of instances with positive gain:', len([x for x in episode_max_gains if x > 0]) / self.batch_size)
        # return torch.mean(self.S).item() - np.mean(episode_gains)
        self.final_value = np.mean(episode_gains)
        # print(episode_max_gain_steps)
        # print(self.bg_list[np.max(episode_max_gain_steps)].kcut_value)
        # print(self.bg_list[0])
        # print(self.bg_list)
        for i in range(self.batch_size):
            bg = self.bg_list[episode_max_gain_steps[i] - 1]
            label_list = bg.ndata["label"][i*k*m:(i+1)*k*m].tolist()
            # print(bg)
            # print(self.bg_list[episode_max_gain_steps[i]])
            self.bg = bg
            print(self.bg.ndata['label'])
            acc, _ = write_result(1, k, m, label_list)
            # print(acc)
            self.acc += acc
        self.acc /= self.batch_size

        return sum(episode_end_value) / len(episode_end_value)

