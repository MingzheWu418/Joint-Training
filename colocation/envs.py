from graph_handler import *

def peek_greedy_reward(states, actions=None, action_type='swap', m=4):
    """
    :param states: LightGraph
    :param actions:
    :param action_type:
    :return:
    """
    batch_size = states.batch_size
    n = states.n
    bn = batch_size * n

    if actions is None:
        actions = get_legal_actions(states=states, action_type=action_type, m=m)
    group_matrix = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                             states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(bn, n)
    num_action = actions.shape[0] // batch_size
    # print(group_matrix)
    action_mask = torch.tensor(range(0, bn, n)).unsqueeze(1).expand(batch_size, 2).repeat(1, num_action).view(
        num_action * batch_size, -1)
    if states.in_cuda:
        actions_ = actions + action_mask.cuda()
    else:
        actions_ = actions + action_mask
    #  (b, n, n)
    #  (b * num_action, n)
    if action_type == 'swap':
        rewards = (states.ndata['adj'][actions_[:, 0], :] * (
                    group_matrix[actions_[:, 0], :] - group_matrix[actions_[:, 1], :]) ).sum(dim=1) \
                  + (states.ndata['adj'][actions_[:, 1], :] * (
                    group_matrix[actions_[:, 1], :] - group_matrix[actions_[:, 0], :])).sum(dim=1) \
                  + 2 * states.ndata['adj'][actions_[:, 0], actions[:, 1]]
    elif action_type == 'flip':
        action_mask_k = torch.tensor(range(0, batch_size * states.k, states.k)).unsqueeze(1).expand(batch_size, 2).repeat(1, num_action).view(
            num_action * batch_size, -1)
        if states.in_cuda:
            group_matrix -= torch.eye(n).repeat(batch_size, 1).cuda()
            actions__ = actions + action_mask_k.cuda()
        else:
            group_matrix -= torch.eye(n).repeat(batch_size, 1)
            actions__ = actions + action_mask_k
        change_group_matrix = states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(-1, n)
        rewards = (states.ndata['adj'][actions_[:, 0], :] * (group_matrix[actions_[:, 0], :] - change_group_matrix[actions__[:, 1], :])).sum(dim=1)

    return rewards

def greedy_solver(graph, step=10):
    Actions = []
    Rewards = []
    # graph = dc(graph)
    m = 3
    for j in range(step):
        actions = get_legal_actions(states=graph, m=m)
        # print(actions)
        r = peek_greedy_reward(states=graph, actions=actions, m=m)
        Rewards.append(r.max().item())
        chosen_action = actions[r.argmax()].unsqueeze(0)
        Actions.append(chosen_action)
        graph, rr = step_batch(states=graph, action=chosen_action)
        # print("New graph after action: ")
        # print(aa)
        # print(rr)
        # print(graph)
    return graph, Actions, Rewards



def get_legal_actions(states, action_type='swap', action_dropout=1.0, pause_action=False, return_num_action=False, m=4):
    """
    :param states: LightGraph
    :param action_type:
    :param action_dropout:
    :param pause_action:
    :param return_num_action: if only returns the number of actions
    :return:
    """
    if action_type == 'flip':

        legal_actions = torch.nonzero(1 - states.ndata['label'])
        num_actions = legal_actions.shape[0] // states.batch_size
        if return_num_action:
            if pause_action:
                return int(num_actions * action_dropout) + 1
            else:
                return int(num_actions * action_dropout)

        mask = torch.tensor(range(0, states.n * states.batch_size, states.n)).repeat(num_actions).view(-1, states.batch_size).t().flatten()
        if states.in_cuda:
            legal_actions[:, 0] -= mask.cuda()
        else:
            legal_actions[:, 0] -= mask

        if action_dropout < 1.0:
            maintain_actions = int(num_actions * action_dropout)
            maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(states.batch_size)]
            legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
        if pause_action:
            legal_actions = legal_actions.reshape(states.batch_size, -1, 2)
            legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0 ).unsqueeze(1)], dim=1).view(-1, 2)

    if action_type == 'swap':

        n = states.n
        # print(states)
        mask = torch.eye(n).repeat(states.batch_size, 1, 1)
        
        mask = torch.bmm(states.ndata['label'].view(states.batch_size, n, -1), # say 10, 40, 10 (equals to k)
                         states.ndata['label'].view(states.batch_size, n, -1).transpose(1, 2)) # 10, 10, 40
                         # ones = in the same room
                         
        legal_actions = torch.triu(1 - mask).nonzero()[:, 1:3]  # tensor (270, 2)
        # we only want [0, 4], [0, 8], [1, 5], [1, 9] etc
        type_constraint = (legal_actions[:, 1] - legal_actions[:, 0])%m == 0
        legal_actions = legal_actions[type_constraint].cuda()
        # print(legal_actions)
        # print(legal_actions.shape)

        num_actions = legal_actions.shape[0] // states.batch_size
        if return_num_action:
            if pause_action:
                return int(num_actions * action_dropout) + 1
            else:
                return int(num_actions * action_dropout)

        if action_dropout < 1.0:
            num_actions = legal_actions.shape[0] // states.batch_size
            maintain_actions = int(num_actions * action_dropout)
            maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(states.batch_size)]
            legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
        if pause_action:
            legal_actions = legal_actions.reshape(states.batch_size, -1, 2)
            legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0).unsqueeze(1)], dim=1).view(-1, 2)

        # Add type constraint because we are using real data, 
        # and the same type of sensor should not be classified in the same room
        # torch.set_printoptions(profile="full")
        # print(legal_actions)
        # print(legal_actions.shape)
        # torch.set_printoptions(profile="default")
    # print("-----")
    return legal_actions


def step_batch(states, action, action_type='swap', return_sub_reward=False):
    """
    :param states: LightGraph
    :param action: torch.tensor((batch_size, 2))
    :return:
    """
    # print(states.batch_size)
    # print(action.shape)
    assert states.batch_size == action.shape[0]

    if states.in_cuda:
        mask = torch.tensor(range(0, states.n * states.batch_size, states.n)).cuda()
    else:
        mask = torch.tensor(range(0, states.n * states.batch_size, states.n))

    batch_size = states.batch_size
    n = states.n

    ii, jj = action[:, 0], action[:, 1]
    # print("===== step =====")
    # print(action)

    old_S = states.kcut_value

    if action_type == 'swap':
        # swap two sets of nodes
        # print("============= BEFORE =============")
        # print(states.ndata['label'])
        tmp = dc(states.ndata['label'][ii + mask])
        states.ndata['label'][ii + mask] = states.ndata['label'][jj + mask]
        states.ndata['label'][jj + mask] = tmp

        # print("============= AFTER  =============")
        # print(states.ndata['label'])
    else:
        # flip nodes
        states.ndata['label'][ii + mask] = torch.nn.functional.one_hot(jj, states.k).float()
    # rewire edges
    nonzero_idx = [i for i in range(states.n ** 2) if i % (states.n + 1) != 0]
    states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                                             states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)) \
                                       .view(batch_size, -1)[:, nonzero_idx].view(-1)
    '''
    print(states.ndata['label'].view(batch_size, n, -1))
    '''
    # compute new S
    new_S = calc_S(states)
    # new_S = calc_S(states)
    # print(new_S)
    states.kcut_value = new_S

    # rewards = torch.sum(old_S - new_S)
    rewards = old_S - new_S
    # print(rewards)
    # rewards = torch.zeros(2,2).requires_grad_()

    return states, rewards


