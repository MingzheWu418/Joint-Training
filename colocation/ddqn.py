from re import M, X
import time
from networks import *
from log_utils import logger
from dataclasses import dataclass
import itertools
from pynvml import *
from losses import tripletLoss, combLoss
import pickle

class EpisodeHistory:
    def __init__(self, g, max_episode_len, action_type='swap'):
        self.action_type = action_type
        self.init_state = dc(g)

        self.n = g.number_of_nodes()
        self.max_episode_len = max_episode_len
        self.episode_len = 0
        self.action_seq = []
        self.action_indices = []
        self.reward_seq = []
        self.q_pred = []
        self.action_candidates = []
        self.enc_state_seq = []
        self.sub_reward_seq = []
        if self.action_type == 'swap':
            self.label_perm = torch.tensor(range(self.n)).unsqueeze(0)
        if self.action_type == 'flip':
            self.label_perm = self.init_state.ndata['label'].nonzero()[:, 1].unsqueeze(0)
        self.best_gain_sofar = 0
        self.current_gain = 0
        self.loop_start_position = 0

    def perm_label(self, label, action):
        label = dc(label)
        if self.action_type == 'swap':
            tmp = dc(label[action[0]])
            label[action[0]] = label[action[1]]
            label[action[1]] = tmp
        if self.action_type == 'flip':
            label[action[0]] = action[1]
        return label.unsqueeze(0)

    def write(self, action, action_idx, reward, q_val=None, actions=None, state_enc=None, sub_reward=None, loop_start_position=None):

        new_label = self.perm_label(self.label_perm[-1, :], action)

        self.action_seq.append(action)

        self.action_indices.append(action_idx)

        self.reward_seq.append(reward)

        self.q_pred.append(q_val)

        self.action_candidates.append(actions)

        self.sub_reward_seq.append(sub_reward)

        self.loop_start_position = loop_start_position

        self.label_perm = torch.cat([self.label_perm, new_label], dim=0)

        self.enc_state_seq.append(state_enc)

        self.episode_len += 1

    def wrap(self):
        self.reward_seq = torch.tensor(self.reward_seq)
        self.empl_reward_seq = torch.tensor(self.empl_reward_seq)
        self.label_perm = self.label_perm.long()

def calc_same_room(label, action, m, k): 
    # We use this function to check the maximum matching number of nodes within a room,
    # after the given action has taken place
    room_assigned = torch.argmax(label, dim=1).reshape(-1,m*k) # 100 * 40
    rooms_changed = torch.cat([room_assigned[i, action[i]] for i in range(len(room_assigned))]).reshape(-1, 2)
    action0_index = torch.where(torch.cat([room_assigned[i] == rooms_changed[i][0] for i in range(rooms_changed.shape[0])]) == True)[0]
    action1_index = torch.where(torch.cat([room_assigned[i] == rooms_changed[i][1] for i in range(rooms_changed.shape[0])]) == True)[0]
    identical_room_action0 = torch.div((action0_index.reshape(-1, m)), m, rounding_mode='floor')
    identical_room_action1 = torch.div((action1_index.reshape(-1, m)), m, rounding_mode='floor')
    
    one_hot_0 = torch.nn.functional.one_hot(identical_room_action0)
    one_hot_1 = torch.nn.functional.one_hot(identical_room_action1)

    count0, _ = torch.max(torch.sum(one_hot_0, 1), 1)
    count1, _ = torch.max(torch.sum(one_hot_1, 1), 1)
    return count0+count1

@dataclass
class sars:
    s0: LightGraph
    i: torch.tensor
    l0: torch.tensor
    a: tuple
    r: float
    s1: LightGraph
    # l1: torch.tensor
    rollout_r: torch.tensor
    rollout_a: torch.tensor


class DQN:
    def __init__(self, graph_generator
                 , hidden_dim=32
                 , action_type='swap'
                 , gamma=1.0, eps=0.1, lr=1e-4, action_dropout=1.0
                 , sample_batch_episode=False
                 , replay_buffer_max_size=5000
                 , epi_len=50, new_epi_batch_size=10
                 , cuda_flag=True
                 , explore_method='epsilon_greedy'
                 , priority_sampling='False'
                 , DML_model=None
                 , model_file=None):
        self.cuda_flag = cuda_flag
        self.graph_generator = graph_generator
        self.gen_training_sample_first = False
        if self.gen_training_sample_first:
            self.training_instances = un_batch(self.graph_generator.generate_graph(batch_size=100, cuda_flag=self.cuda_flag), copy=False)
        self.action_type = action_type
        self.m = graph_generator.m
        self.k = graph_generator.k
        self.ajr = graph_generator.ajr
        self.hidden_dim = hidden_dim  # hidden dimension for node representation
        self.n = graph_generator.n
        print("----")
        print(self.m)
        print(self.k)
        print(self.ajr)
        self.eps = eps  # constant for exploration in dqn
        assert explore_method in ['epsilon_greedy', 'softmax', 'soft_dqn']
        self.explore_method = explore_method
        if cuda_flag:
            self.model = DQNet(k=self.k, n=self.n, hidden_dim=self.hidden_dim).cuda()
        else:
            self.model = DQNet(k=self.k, n=self.n, hidden_dim=self.hidden_dim)
        if model_file is not None:
            self.model = pickle.load(open(model_file, 'rb'))
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.model = torch.nn.DataParallel(self.model)
        # print(self.model)
        self.model_target = dc(self.model)
        self.gamma = gamma  # reward decay const
        self.DML_model = DML_model
        self.fine_tune_optimizer = torch.optim.Adam(self.DML_model.parameters(), lr = 3e-4, weight_decay=1e-2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = torch.optim.Adam([p[1] for p in filter(lambda p: p[0] in ['L1.weight', 'L2.weight', 'L1.bias', 'L2.bias'], self.model.named_parameters())], lr=lr)
        self.sample_batch_episode = sample_batch_episode
        self.experience_replay_buffer = []
        self.replay_buffer_max_size = replay_buffer_max_size
        self.buf_epi_len = epi_len  # 50
        self.new_epi_batch_size = new_epi_batch_size  # 10
        self.cascade_replay_buffer = [[] for _ in range(self.buf_epi_len)]
        self.cascade_replay_buffer_weight = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.stage_max_sizes = [self.replay_buffer_max_size // self.buf_epi_len] * self.buf_epi_len  # [100, 100, ..., 100]
        # self.stage_max_sizes = list(range(100,100+4*50, 4))
        self.buffer_actual_size = sum(self.stage_max_sizes)
        self.priority_sampling = priority_sampling
        self.cascade_buffer_kcut_value = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.action_dropout = action_dropout
        self.log = logger()
        self.Q_err = 0  # Q error
        self.triplet_loss = 0
        self.log.add_log('tot_return')
        self.log.add_log('Q_error')
        self.log.add_log('Triplet_error')
        self.log.add_log('Reconstruction_error')
        self.log.add_log('Act_Prop_error')
        self.log.add_log('entropy')
        self.log.add_log('R_signal_posi_len')
        self.log.add_log('R_signal_nega_len')
        self.log.add_log('R_signal_posi_mean')
        self.log.add_log('R_signal_nega_mean')
        self.log.add_log('R_signal_nonzero')
        self.log.add_log('S_new_training_sample')

    def _updata_lr(self, step, max_lr, min_lr, decay_step):
        for g in self.optimizer.param_groups:
            g['lr'] = max(max_lr / ((max_lr / min_lr) ** (step / decay_step) ), min_lr)

    def run_batch_episode(self, target_bg=None, action_type='swap', gnn_step=3, episode_len=50, batch_size=10, rollout_step=1, raw_x=None):
        # This method is used at the beginning of each trial, 
        # in order to fill the buffer with enough experience to start
        sum_r = 0

        if target_bg is None:
            if self.gen_training_sample_first:
                bg = make_batch(np.random.choice(self.training_instances, batch_size, replace=False))
            else:
                bg = self.graph_generator.generate_graph(batch_size=batch_size, cuda_flag=self.cuda_flag)
            self.log.add_item('S_new_training_sample', torch.mean(bg.kcut_value).item())
       
        else:
            assert target_bg.in_cuda == self.cuda_flag
            x = self.DML_model(raw_x.cuda())
            x = x.detach().cpu()
            new_label = target_bg.ndata['label'].clone().detach().cpu()
            new_index = target_bg.ndata['index'].clone().detach().cpu()
            bg = self.graph_generator.generate_graph(x=x, index=new_index, batch_size=batch_size, cuda_flag=self.cuda_flag, label_input=new_label)
            perm_weight(bg)
            del x
            torch.cuda.empty_cache()
        

        num_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout, return_num_action=True, m=self.m)

        action_mask = torch.tensor(range(0, num_actions * batch_size, num_actions))
        if self.cuda_flag:
            action_mask = action_mask.cuda()

        explore_dice = (torch.rand(episode_len, batch_size) < self.eps)
        explore_replace_mask = explore_dice.nonzero()
        explore_step_offset = torch.cat([torch.zeros([1], dtype=torch.long), torch.cumsum(explore_dice.sum(dim=1), dim=0)], dim=0)
        explore_replace_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], ))
        if self.cuda_flag:
            explore_replace_actions = explore_replace_actions.cuda()

        t = 0
        while t < episode_len:

            batch_legal_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout, m=self.m)

            # epsilon greedy strategy
            _, _, _, Q_sa = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)
            
            best_actions = Q_sa.view(-1, num_actions).argmax(dim=1)
            explore_episode_indices = explore_replace_mask[explore_step_offset[t]: explore_step_offset[t + 1]][:, 1]
            explore_actions = explore_replace_actions[explore_step_offset[t]: explore_step_offset[t + 1]]
            best_actions[explore_episode_indices] = explore_actions

            best_actions += action_mask

            actions = batch_legal_actions[best_actions]

            # update bg inplace and calculate batch rewards
            g0 = [g for g in un_batch(bg)]  # current_state
            _, rewards = step_batch(states=bg, action=actions, action_type=action_type)
            g1 = [g for g in un_batch(bg)]  # after_state

            _rollout_reward = torch.zeros((rollout_step))
            _rollout_action = torch.zeros((rollout_step*2)).int()
            if self.cuda_flag:
                _rollout_reward = _rollout_reward.cuda()
                _rollout_action = _rollout_action.cuda()
            if self.sample_batch_episode:
                # self.experience_replay_buffer.extend([sars(g0[i].ndata['index'], g0[i].ndata['label'], actions[i], rewards[i], g1[i].ndata['label'], _rollout_reward, _rollout_action) for i in range(batch_size)])
                self.experience_replay_buffer.extend([sars(g0[i], g0[i].ndata['index'], g0[i].ndata['label'], actions[i], rewards[i], g1[i], _rollout_reward, _rollout_action) for i in range(batch_size)])
            else:  # using cascade buffer
                # self.cascade_replay_buffer[t].extend([sars(g0[i].ndata['index'], g0[i].ndata['label'], actions[i], rewards[i], g1[i].ndata['label'], _rollout_reward, _rollout_action) for i in range(batch_size)])
                self.cascade_replay_buffer[t].extend([sars(g0[i], g0[i].ndata['index'], g0[i].ndata['label'], actions[i], rewards[i], g1[i], _rollout_reward, _rollout_action) for i in range(batch_size)])

                if self.priority_sampling:
                    # compute prioritized weights
                    batch_legal_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout, m=self.m)
                    _, _, _, Q_sa_next = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)
                    delta = Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
                    # delta = (Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / (torch.clamp(torch.abs(Q_sa[best_actions]),0.1))
                    self.cascade_replay_buffer_weight[t, :batch_size] = torch.abs(delta.detach())
            R = [reward.item() for reward in rewards]
            sum_r += sum(R)

            t += 1

        self.log.add_item('tot_return', sum_r)

        return R

    def soft_target(self, Q_sa, batch_size, Temperature=0.1):
        mean_Q_sa = torch.mean(Q_sa.view(batch_size, -1), dim=1)
        return torch.log(torch.mean(torch.exp((Q_sa.view(batch_size, -1) - mean_Q_sa.unsqueeze(1)).clamp(-8, 8) / Temperature),
                             dim=1)) * Temperature + mean_Q_sa

    def sample_actions_from_q(self, prop_a_score, Q_sa, batch_size, Temperature=1.0, eps=None, top_k=1):
        num_actions = Q_sa.shape[0] // batch_size
        if self.explore_method == 'epsilon_greedy':
            # len = batch_size * topk  (g0_top1, g1_top1, ..., g0_top2, ...)
            best_actions = Q_sa.view(batch_size, num_actions).topk(k=top_k, dim=1).indices.t().flatten()

            # update prop_net
            b = best_actions.shape[0]
            L = -torch.log(prop_a_score[range(b), best_actions]+1e-8) + 1.0 * (torch.log(prop_a_score+1e-8) * prop_a_score).sum(dim=1)
            L_sum = L.sum()
            L_sum.backward(retain_graph=True)

            self.optimizer2.step()
            self.optimizer2.zero_grad()

            self.log.add_item('Act_Prop_error', L_sum.item())

        if self.explore_method == 'softmax' or self.explore_method == 'soft_dqn':

            best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, num_actions) / Temperature), 1).view(-1)

        if eps is None:
            eps = self.eps
        explore_replace_mask = (torch.rand(batch_size * top_k) < eps).nonzero()
        explore_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], ))
        if self.cuda_flag:
            explore_actions = explore_actions.cuda()
        best_actions[explore_replace_mask[:, 0]] = explore_actions
        # add action batch offset
        if self.cuda_flag:
            best_actions += torch.tensor(range(0, num_actions * batch_size, num_actions)).repeat(top_k).cuda()
        else:
            best_actions += torch.tensor(range(0, num_actions * batch_size, num_actions)).repeat(top_k)
        return best_actions

    def rollout(self, bg, rollout_step, top_num=5):

        # batch_size = self.new_epi_batch_size * self.buf_epi_len * top_num
        batch_size = bg.batch_size
        rollout_rewards = torch.zeros((batch_size, rollout_step))
        rollout_actions = torch.zeros((batch_size, rollout_step * 2)).int()
        if self.cuda_flag:
            rollout_rewards = rollout_rewards.cuda()
            rollout_actions = rollout_actions.cuda()

        for step in range(rollout_step):
            batch_legal_actions = get_legal_actions(states=bg, action_type=self.action_type, action_dropout=self.action_dropout, m=self.m)

            prop_a_score, prop_actions = self.model.forward_prop(bg, batch_legal_actions, action_type=self.action_type)

            _, _, _, Q_sa = self.model(bg, prop_actions, action_type=self.action_type)

            chosen_actions = self.sample_actions_from_q(prop_a_score, Q_sa, batch_size, eps=0.0, top_k=1)
            _actions = prop_actions[chosen_actions]

            # update bg inplace and calculate batch rewards
            _, _rewards = step_batch(states=bg, action_type=self.action_type, action=_actions)

            rollout_rewards[:, step] = _rewards
            rollout_actions[:, 2 * step:2 * step + 2] = _actions
            # print('step', step, rewards)
        return rollout_rewards, rollout_actions

    def run_cascade_episode(self, target_bg=None, action_type='swap', gnn_step=3, rollout_step=0, verbose=False, epoch=None, raw_x=None):
        # this method runs one step of the episode, it samples data from 
        sum_r = 0

        T0 = time.time()
        
        # generate new start states
        if target_bg is None:
            if self.gen_training_sample_first:
                new_graphs = make_batch(np.random.choice(self.training_instances, self.new_epi_batch_size, replace=False))
            else:
                new_graphs = self.graph_generator.generate_graph(batch_size=self.new_epi_batch_size, cuda_flag=self.cuda_flag)

            if epoch is not None:
                # shift the starting point of an episode
                self.rollout(bg=new_graphs, rollout_step=epoch // 2000)

            new_graphs = un_batch(new_graphs, copy=False)
            self.log.add_item('S_new_training_sample', torch.mean(torch.cat([new_graphs[i].kcut_value for i in range(self.new_epi_batch_size)])).item())
        
        else:
            assert target_bg.in_cuda == self.cuda_flag
            new_label = target_bg.ndata['label'].detach().cpu()
            new_index = target_bg.ndata['index'].detach().cpu()
            x = self.DML_model(raw_x.cuda())
            x = x.detach().cpu()
            new_graphs = self.graph_generator.generate_graph(x=x, index=new_index, batch_size=self.new_epi_batch_size, cuda_flag=self.cuda_flag, label_input=new_label)
            
            # perm_weight(new_graphs)
            # new_graphs = un_batch(new_graphs, copy=False)
            new_graphs = un_batch(new_graphs, copy=True)
        
        
            
        if verbose:
            T1 = time.time(); print('t1', T1 - T0)

        # extend previous states(no memory copy here)
        # TODO: implement this part
        # for i in range(self.buf_epi_len-1):
        #     for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]:
        #         g0 = self.graph_generator.generate_graph(x=x, index=tpl.i, batch_size=1, cuda_flag=self.cuda_flag, label_input=tpl.l0) 
        #         new_graph, rewards = step_batch(g0, tpl.a.unsqueeze(0))
        #         new_graphs.append(new_graph)
        # bg = make_batch(new_graphs)

        # optimized 0.25s per iter 
        action_list = torch.cat([torch.cat([tpl.a.unsqueeze(0) for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]], 1) for i in range(self.buf_epi_len-1)], 0).reshape(-1, 2)
        index_list = torch.cat([torch.cat([tpl.i for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]]) for i in range(self.buf_epi_len-1)])
        label_list = torch.cat([torch.cat([tpl.l0 for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]]) for i in range(self.buf_epi_len-1)])
        g0 = self.graph_generator.generate_graph(x=x, index=index_list, batch_size=self.new_epi_batch_size*(self.buf_epi_len-1), cuda_flag=self.cuda_flag, label_input=label_list) 

        new_graph, rewards = step_batch(g0, action_list)
        
        new_graphs.extend(un_batch(new_graph))

        if verbose:
            T2 = time.time(); print('t2', T2 - T1)

        # make batch and copy new states
        bg = make_batch(new_graphs)
        if verbose:
            T3 = time.time(); print('t3', T3 - T2)

        batch_size = self.new_epi_batch_size * self.buf_epi_len

        if verbose:
            T4 = time.time(); print('t4', T4 - T3)

        batch_legal_actions = get_legal_actions(states=bg, action_type=self.action_type, action_dropout=self.action_dropout, m=self.m)
        if verbose:
            T5 = time.time(); print('t5', T5 - T4)
        # epsilon greedy strategy
        # TODO: multi-gpu parallelization

        prop_a_score, prop_actions = self.model.forward_prop(bg, batch_legal_actions, action_type=self.action_type)
        _, _, _, Q_sa = self.model(bg, prop_actions, action_type=self.action_type)

        if verbose:
            T6 = time.time(); print('t6', T6 - T5)

        if not rollout_step:
            # TODO: can alter explore strength according to kcut_valueS
            chosen_actions = self.sample_actions_from_q(prop_a_score, Q_sa, batch_size, Temperature=self.eps)
            actions = prop_actions[chosen_actions]
            rollout_rewards = torch.zeros(batch_size, 1)
            rollout_actions = torch.zeros(batch_size, 2).int()
        else:
            top_num = 1  # rollout for how many top actions
            rollout_bg = make_batch([bg] * top_num)

            # chosen_actions - len = batch_size * topk
            chosen_actions = self.sample_actions_from_q(prop_a_score, Q_sa, batch_size, Temperature=self.eps, top_k=top_num)

            topk_actions = prop_actions[chosen_actions]

            bg1, rewards1 = step_batch(states=rollout_bg, action_type=action_type, action=topk_actions)

            rollout_rewards, rollout_actions = self.rollout(bg=bg1, rollout_step=rollout_step, top_num=top_num)

            # select actions based on rollout rewards
            # rollout_selected_actions = torch.cat([rewards1.view(-1, 1), rollout_rewards], dim=1)\
            rollout_selected_actions = torch.cat([rollout_rewards], dim=1)\
                .cumsum(dim=1).max(dim=1)\
                .values.view(top_num, -1)\
                .argmax(dim=0) * batch_size
            if self.cuda_flag:
                rollout_selected_actions += torch.tensor(range(batch_size)).cuda()
            else:
                rollout_selected_actions += torch.tensor(range(batch_size))

            # update bg inplace and calculate batch rewards
            actions = topk_actions[rollout_selected_actions, :]
            # rewards = rewards1[rollout_selected_actions]
            rollout_rewards = rollout_rewards[rollout_selected_actions, :]
            rollout_actions = rollout_actions[rollout_selected_actions, :]

        # update bg inplace and calculate batch rewards
        _, rewards = step_batch(states=bg, action_type=action_type, action=actions)

        g0 = new_graphs  # current_state
        g1 = un_batch(bg, copy=False)  # after_state

        [self.cascade_replay_buffer[t].extend(
            [sars(g0[j+t*self.new_epi_batch_size]
            , g0[j+t*self.new_epi_batch_size].ndata['index']
            , g0[j+t*self.new_epi_batch_size].ndata['label']
            , actions[j+t*self.new_epi_batch_size]
            , rewards[j+t*self.new_epi_batch_size] #+ (4/5)**t
            , g1[j+t*self.new_epi_batch_size]
            # , g1[j+t*self.new_epi_batch_size].ndata['label']
            , rollout_rewards[j+t*self.new_epi_batch_size, :]
            , rollout_actions[j+t*self.new_epi_batch_size, :])
            for j in range(self.new_epi_batch_size)])
         for t in range(self.buf_epi_len)]

        if self.priority_sampling:
            # compute prioritized weights
            batch_legal_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout, m=self.m)
            _, _, _, Q_sa_next = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

            delta = Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
            # delta = (Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / torch.clamp(torch.abs(Q_sa[chosen_actions]),0.1)
            self.cascade_replay_buffer_weight = torch.cat([self.cascade_replay_buffer_weight, torch.abs(delta.detach().cpu().view(self.buf_epi_len, self.new_epi_batch_size))], dim=1).detach()
            # print(self.cascade_replay_buffer_weight)

        R = [reward.item() for reward in rewards]
        sum_r += sum(R)

        self.log.add_item('tot_return', sum_r)
        x = x.cpu()
        del x
        torch.cuda.empty_cache()


        return R

    def sample_from_buffer(self, batch_size, q_step, gnn_step, x=None):

        batch_size = min(batch_size, len(self.experience_replay_buffer))
        
        sample_buffer = np.random.choice(self.experience_replay_buffer, batch_size, replace=False)
        # make batches
        # TODO: finish this part
        batch_begin_state = []
        batch_end_state = []
        for tpl in sample_buffer:
            batch_begin_state.append(self.graph_generator.generate_graph(x=x, index=tpl.i, batch_size=1, cuda_flag=self.cuda_flag, label_input=tpl.l0))
            batch_end_state.append(self.graph_generator.generate_graph(x=x, index=tpl.i, batch_size=1, cuda_flag=self.cuda_flag, label_input=tpl.l1))
        batch_begin_state = make_batch(batch_begin_state)
        batch_end_state = make_batch(batch_end_state)
        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        batch_end_action = get_legal_actions(state=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout, m=self.m)
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

        # only compute limited number for Q_s1a
        # TODO: multi-gpu parallelization
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)


        q = self.gamma ** q_step * Q_s2a.view(-1, action_num).max(dim=1).values - Q_s1a_
        Q = q.unsqueeze(0)

        return torch.cat(R), Q

    def sample_from_cascade_buffer(self, batch_size, q_step, gnn_step, rollout_step=0, raw_x=None):

        batch_size = min(batch_size, len(self.cascade_replay_buffer[0]) * self.buf_epi_len)

        batch_sizes = [
            min(batch_size * self.stage_max_sizes[i] // self.buffer_actual_size, len(self.cascade_replay_buffer[0]))
            for i in range(self.buf_epi_len)]
        
        sample_buffer = list(itertools.chain(*[np.random.choice(a=self.cascade_replay_buffer[i]
                                                            , size=batch_sizes[i]
                                                            , replace=False
                                                            ) for i in range(self.buf_epi_len)]))

        x = self.DML_model(raw_x.cuda()) # calculated embedding given by metric learning

        # initializing the graphs, 
        # we do this because we cannot reuse the graph for RL model's update,
        # or that would lead the metric learning model to collapsing to a trivial solution
        index = torch.cat([tpl.i for tpl in sample_buffer])
        l0_list = torch.cat([tpl.l0 for tpl in sample_buffer])
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer])
        graph = self.graph_generator.generate_graph(x=x.detach(), index=index, batch_size=len(sample_buffer), cuda_flag=self.cuda_flag, label_input=l0_list)
        s0 = [g for g in un_batch(graph)]

        # optimize speed by making into batches
        batch_end_state, rewards = step_batch(states=graph, action=batch_begin_action)
        batch_begin_state = make_batch(s0)
        R = rewards

        # torch.set_printoptions(profile="full")
        begin_label = batch_begin_state.ndata['label'].cuda()
        end_label = batch_end_state.ndata['label'].cuda()
        m = self.m
        k = self.k

        # Correct nodes before action
        room_assigned = torch.argmax(begin_label, dim=1).reshape(-1,m*k) # 100 * 40
        rooms_changed = torch.cat([room_assigned[i, batch_begin_action[i]] for i in range(len(room_assigned))]).reshape(-1, 2)

        # Room assigned at the current state
        # These two lines are buggy bc room_assigned does not match the ground truth room
        # check room_assigned or rooms_changed.
        action0_index = torch.where(torch.cat([room_assigned[i] == rooms_changed[i][0] for i in range(rooms_changed.shape[0])]) == True)[0] # 400
        action1_index = torch.where(torch.cat([room_assigned[i] == rooms_changed[i][1] for i in range(rooms_changed.shape[0])]) == True)[0] # 400
        identical_room_action0 = torch.div((action0_index.reshape(-1, m)), m, rounding_mode='floor')
        identical_room_action1 = torch.div((action1_index.reshape(-1, m)), m, rounding_mode='floor')

        # Correct nodes after action
        count_after_action = calc_same_room(end_label, batch_begin_action, m, k)
        
        # Find ground truth best action exhaustively
        gt_reward = []
        for j in range(m):
            for l in range(m):
                reward_calc0 = identical_room_action0.clone()
                reward_calc1 = identical_room_action1.clone()
                reward_calc0[:, j] = identical_room_action1[:, l]
                reward_calc1[:, l] = identical_room_action0[:, j]
                one_hot_0 = torch.nn.functional.one_hot(reward_calc0)
                one_hot_1 = torch.nn.functional.one_hot(reward_calc1)
                count0, _ = torch.max(torch.sum(one_hot_0, 1), 1)
                count1, _ = torch.max(torch.sum(one_hot_1, 1), 1)
                gt_reward.append(count0+count1)
        gt_reward, action_indices = torch.max(torch.stack(gt_reward), dim=0)
        j_index = torch.div((action_indices), m, rounding_mode='floor').reshape(-1, 1)
        k_index = (action_indices%m+m).reshape(-1, 1)
        gt_action_index = torch.cat([j_index, k_index], dim=1)

        sensor_in_two_clusters = torch.cat([action0_index.reshape(-1, m), action1_index.reshape(-1, m)], dim=1)
        """ 
        Triplet generation if DQN's action is suboptimal
        For every node that is not changed, we set it to be an anchor
        """
        dqn_action_mask = torch.cat((batch_begin_action[:, 0].unsqueeze(1).repeat(1,m), batch_begin_action[:, 1].unsqueeze(1).repeat(1,m)), dim=1)\
                +(torch.arange(batch_begin_action.shape[0]).unsqueeze(1).repeat(1,2*m)*m*k).cuda()
        gt_action_mask = torch.sum(torch.nn.functional.one_hot(gt_action_index), dim=1)
        gt_better = torch.repeat_interleave((gt_reward>count_after_action).unsqueeze(1), 2*m, dim=1)
        # Choose which triplets are used to calculate triplet loss
        # TODO: Questions: 
        # 1. How to identify the error of DQN from error of DML? 
        #   Current solution is train DQN to a level that we believe DQN is good enough

        # Use only the sensors that is not affected by both actions as anchor

        # if sensor_in_two_clusters != dqn_action_mask, meaning the node is not selected by dqn
        # if gt_action_mask < 1, meaning the node is not selected by the ground truth
        anchor_mask = torch.logical_and(torch.logical_and(gt_action_mask < 1, sensor_in_two_clusters != dqn_action_mask), gt_better).reshape(-1,)
        anchor_index = sensor_in_two_clusters.reshape(-1,)[anchor_mask]
        """
        For every node that is chosen to be pulled closer to the anchor by the ground truth action,
        we set it to be a positive node
        """
        pos_mask = gt_action_index.clone()
        pos_mask[:, 0] = gt_action_index[:, 1]
        pos_mask[:, 1] = gt_action_index[:, 0]
        pos_mask = (pos_mask + 2*m*torch.arange(len(pos_mask)).unsqueeze(1).repeat(1,2).cuda()).reshape(-1, 1)
        pos_index = torch.repeat_interleave(sensor_in_two_clusters.reshape(-1, 1)[pos_mask], m, dim=1).reshape(-1,)[anchor_mask]
        """
        For every node that is pulled closer to the anchor by the DQN's choice, but not the ground truth action,
        we set it to be a negative node
        """
        neg_mask = batch_begin_action.clone()
        neg_mask[:,0] = batch_begin_action[:,1]
        neg_mask[:,1] = batch_begin_action[:,0]

        neg_index = neg_mask + (k*m*torch.arange(len(neg_mask)).unsqueeze(1)).repeat(1,2).cuda()
        neg_index = torch.repeat_interleave(neg_index, m, dim=1).reshape(-1,)[anchor_mask]

        # If positive and negative are in the same room, it does not make sense to pull one closer and push another farther
        # If negative and anchor are in the same room, it does not make sense to push negative farther
        # If positive and anchor are not in the same room, it does not make sense to pull positive closer
        pos_room = torch.div(pos_index, m, rounding_mode='floor')
        neg_room = torch.div(neg_index, m, rounding_mode='floor')
        anchor_room = torch.div(anchor_index, m, rounding_mode='floor')
        update_index = torch.logical_and(pos_room != neg_room, anchor_room == pos_room)

        # we add the type constraint here so we do not need to do it in the DQNet class, this is slightly slower but easier to implement
        type_constraint = torch.logical_and(anchor_index%m != neg_index%m, pos_index%m != neg_index%m) 
        use_index = torch.logical_and(update_index, type_constraint)

        anchor_index = anchor_index[use_index]
        pos_index = pos_index[use_index]
        neg_index = neg_index[use_index]

        # this is where we update the metric learning model, 
        # given the current learned embedding and the selected triplets to fine-tune
        embedding = x.reshape(-1, m, 86)[index, :, :].reshape(-1, 86) # entire embedding comes to 4000*805. select 40 * 805 for every graph
        anchor = embedding[anchor_index]
        pos = embedding[pos_index]
        neg = embedding[neg_index]
        criterion = combLoss(margin = 1.0).cuda() # The larger the stricter
        triplet_loss, triplet_correct = criterion(anchor, pos, neg)
        self.log.add_item('Triplet_error', triplet_loss.detach().cpu().numpy())
        # R = torch.cat([tpl.r.unsqueeze(0) for tpl in sample_buffer])
        if self.cuda_flag:
            R = R.cuda()

        if rollout_step:
            rollout_R = torch.cat([R.unsqueeze(1), torch.cat([tpl.rollout_r.unsqueeze(0) for tpl in sample_buffer])], dim=1)
            R = rollout_R[:, :q_step].sum(dim=1)
            rollout_A = torch.cat([tpl.rollout_a.unsqueeze(0) for tpl in sample_buffer]) #[1000,2], or [1,2]*1000
            step_batch(states=batch_end_state, action=rollout_A[:, 0:2 * (q_step - 1)], action_type=self.action_type)
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        _, reconstruct_S, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)

        #  foward the end state for (q_step - 1) steps

        batch_end_action = get_legal_actions(states=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout, m=self.m)

        #  action proposal network
        prop_a_score, prop_actions = self.model_target.forward_prop(batch_end_state, batch_end_action, action_type=self.action_type)

        _, _, _, Q_s2a = self.model_target(batch_end_state, prop_actions, action_type=self.action_type, gnn_step=gnn_step)

        chosen_actions = self.sample_actions_from_q(prop_a_score, Q_s2a, batch_size, Temperature=self.eps)
        # chosen_actions = self.sample_actions_from_q(prop_a_score, Q_s2a, batch_size, Temperature=-torch.log(self.q_err))

        q = self.gamma ** q_step * Q_s2a[chosen_actions].detach() - Q_s1a_

        Q = q.unsqueeze(0)

        x = x.detach()
        return R, Q, triplet_loss, reconstruct_S

    def back_loss(self, R, Q, triplet_loss, err_S, update_model=True, epoch=0, fine_tune=False):
        beta = 30.0
        R = R.cuda(device=Q.device)
        L_dqn = torch.pow(R + Q, 2).sum() + triplet_loss
        L = L_dqn
        L.backward(retain_graph=False)
        self.Q_err += L_dqn.item()
        if update_model:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if fine_tune:
                self.fine_tune_optimizer.step()
                self.fine_tune_optimizer.zero_grad()
            self.log.add_item('Q_error', self.Q_err)
            # self.log.add_item('Reconstruction_error', 0)
            self.Q_err = 0
            self.log.add_item('entropy', 0)

    def train_dqn(self, target_bg=None, epoch=0, batch_size=16, num_episodes=10, episode_len=50, gnn_step=10, q_step=1, grad_accum=1, rollout_step=0, ddqn=False, raw_x=None, fine_tune=False):
        """
        :param batch_size:
        :param num_episodes:
        :param episode_len: #steps in each episode
        :param gnn_step: #iters when running gnn
        :param q_step: reward delay step
        :param ddqn: train in ddqn mode
        :return:
        """
        if self.sample_batch_episode:
            T3 = time.time()
            self.run_batch_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=episode_len,
                                   batch_size=num_episodes)
            T4 = time.time()

            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            R, Q = self.sample_from_buffer(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step)

            T6 = time.time()
        else:
            T3 = time.time()

            # with torch.autograd.set_detect_anomaly(True):
            if epoch == 0:
                # buf_epi_len calls of model(new_epi_batch_size) * action_num
                self.run_batch_episode(target_bg=target_bg, action_type=self.action_type, gnn_step=gnn_step, episode_len=self.buf_epi_len,
                                batch_size=self.new_epi_batch_size, rollout_step=rollout_step, raw_x=raw_x)
            else:
                # 1 call of model(buf_epi_len * new_epi_batch_size) * action_num
                self.run_cascade_episode(target_bg=target_bg, action_type=self.action_type, gnn_step=gnn_step, rollout_step=rollout_step, epoch=None, raw_x=raw_x)
            T4 = time.time()
            # trim experience replay buffer
            self.trim_replay_buffer(epoch)
            R, Q, triplet_loss, err_S = self.sample_from_cascade_buffer(batch_size=batch_size, q_step=q_step, rollout_step=rollout_step, gnn_step=gnn_step, raw_x=raw_x)
            T6 = time.time()


            for _ in range(grad_accum - 1):
                self.back_loss(R, Q, triplet_loss, err_S, update_model=False, epoch=epoch, fine_tune=False)
                del R, Q, triplet_loss, err_S
                torch.cuda.empty_cache()
                R, Q, triplet_loss, err_S = self.sample_from_cascade_buffer(batch_size=batch_size, q_step=q_step, rollout_step=rollout_step,
                                                    gnn_step=gnn_step, raw_x=raw_x)
            self.back_loss(R, Q, triplet_loss, err_S, update_model=True, epoch=epoch, fine_tune=fine_tune)
            triplet_loss = triplet_loss.detach().cpu()
            del R, Q, triplet_loss, err_S
            torch.cuda.empty_cache()
            T7 = time.time()

            self._updata_lr(step=epoch, max_lr=2e-3, min_lr=1e-3, decay_step=10000)

            print('Rollout time:', T4-T3)
            print('Sample and forward time', T6-T4)
            print('Backloss time', T7-T6)
        
        return self.log

    def trim_replay_buffer(self, epoch):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]

        if epoch * self.buf_epi_len * self.new_epi_batch_size > self.replay_buffer_max_size:
            for i in range(self.buf_epi_len):
                self.cascade_replay_buffer[i] = self.cascade_replay_buffer[i][-self.stage_max_sizes[i]:]

    def update_target_net(self):
        self.model_target.load_state_dict(self.model.state_dict())
