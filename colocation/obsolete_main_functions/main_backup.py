import argparse
import torch
import torch.nn.functional as F
import sys
import os
import random
import utils.util as utils
import numpy as np
import pandas as pd
import datetime
import statistics
import pickle

from Data import *
from models import STN
from losses import tripletLoss, combLoss
import scipy.io as scio
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
from GeneticAlgorithm.colocation import run
from scipy.spatial.distance import euclidean
import csv
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from sklearn.manifold import MDS, TSNE
import graph_handler
from ddqn import DQN
from graph_handler import GraphGenerator
from validation import test_summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from GeneticAlgorithm.colocation.core import corr_score
from GeneticAlgorithm.colocation.optimizers import strict_genetic_algorithm as ga

from fastdtw import fastdtw

from pytorch_metric_learning.losses import TripletMarginLoss, AngularLoss, MultipleLosses, CentroidTripletLoss, SupConLoss
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance, SNRDistance
from pytorch_metric_learning.miners import TripletMarginMiner, BatchHardMiner, PairMarginMiner
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from scipy.spatial import distance_matrix

from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise, Dropout, Pool

import imageio
from pynvml import *
# import psutil
# import gc

# import sys
# np.set_printoptions(threshold=sys.maxsize)
# torch.set_printoptions(profile="full")
# torch.cuda.set_device(0)

cuda_flag = True
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
save_log = True

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-config', default = 'stn', type =str)
    parser.add_argument('-model', default='stn', type=str,
                        choices=['stn'])
    parser.add_argument('-loss', default='triplet', type=str,
                        choices=['triplet', 'comb'])
    parser.add_argument('-seed', default=42, type=int,
                        help="Random seed") # seed 42 results in a bad test graph. Can be used to test end-to-end training
    parser.add_argument('-log', default='stn', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default=10606, type=int,
                        help="Log directory")
    parser.add_argument('-split',default='room', type=str,
                        help="split 1/5 sensors or rooms for test",
                        choices = ['room', 'sensor'])
    parser.add_argument('--save_folder', default='/test')
    parser.add_argument('--train_distr', default='cluster', help="")
    parser.add_argument('--test_distr', default='cluster', help="")
    parser.add_argument('--target_mode', default=False)
    parser.add_argument('--k', default=10, help="size of K-cut")
    parser.add_argument('--m', default='4', help="cluster size")
    parser.add_argument('--ajr', default=39, help="")
    parser.add_argument('--h', default=128, help="hidden dimension")
    parser.add_argument('--rollout_step', default=1)
    parser.add_argument('--q_step', default=2)
    parser.add_argument('--batch_size', default=100, help='')
    parser.add_argument('--n_episode', default=10, help='')
    parser.add_argument('--episode_len', default=100, help='')
    parser.add_argument('--grad_accum', default=1, help='')
    parser.add_argument('--action_type', default='swap', help="")
    parser.add_argument('--gnn_step', default=3, help='')
    parser.add_argument('--test_batch_size', default=1, help='')
    parser.add_argument('--validation_step', default=200, help='')
    parser.add_argument('--gpu', default='1', help="")
    parser.add_argument('--resume', default=False)
    parser.add_argument('--problem_mode', default='complete', help="")
    parser.add_argument('--readout', default='mlp', help="")
    parser.add_argument('--edge_info', default='adj_dist')
    parser.add_argument('--clip_target', default=0)
    parser.add_argument('--explore_method', default='epsilon_greedy')
    parser.add_argument('--priority_sampling', default=0)
    parser.add_argument('--gamma', type=float, default=0.9, help="")
    parser.add_argument('--eps0', type=float, default=0.5, help="") # 0.5
    parser.add_argument('--eps', type=float, default=0.1, help="") # 0.1
    parser.add_argument('--explore_end_at', type=float, default=0.3, help="") # 0.3
    parser.add_argument('--anneal_frac', type=float, default=0.7, help="") # 0.7
    # 0.5 -> 0.1 -> 0.0
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--action_dropout', type=float, default=1.0)
    parser.add_argument('--n_epoch', default=2001)
    parser.add_argument('--save_ckpt_step', default=20000)
    parser.add_argument('--target_update_step', default=5)
    parser.add_argument('--replay_buffer_size', default=5000, help="") 
    parser.add_argument('--sample_batch_episode', type=int, default=0, help='')
    parser.add_argument('--ddqn', default=False)
    args = parser.parse_args()
    config = utils.read_config(args.config + '.yaml')
    return args, config

args, config = parse_args()

try:
    random.seed(args.seed)
    np.random.seed(args.seed)
except:
    print("No seed available")

gpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_folder = args.save_folder
resume = args.resume
target_mode = args.target_mode
problem_mode = args.problem_mode
readout = args.readout
action_type = args.action_type
k = int(args.k)
m = [int(i) for i in args.m.split(',')]
if len(m) == 1:
    m = m[0]
    N = k * m
else:
    N = sum(m)
if k == 3 and m == 4:
    run_validation_33 = True
else:
    run_validation_33 = False
ajr = int(args.ajr)
train_graph_style = args.train_distr
test_graph_style = args.test_distr
h = int(args.h)
edge_info = args.edge_info
clip_target = bool(int(args.clip_target))
explore_method = args.explore_method
priority_sampling = bool(int(args.priority_sampling))
gamma = float(args.gamma)
lr = args.lr    # learning rate
action_dropout = args.action_dropout
replay_buffer_size = int(args.replay_buffer_size)
target_update_step = int(args.target_update_step)
batch_size = int(args.batch_size)
grad_accum = int(args.grad_accum)
sample_batch_episode = bool(args.sample_batch_episode)
n_episode = int(args.n_episode)
test_episode = int(args.test_batch_size)
validation_step = int(args.validation_step)
episode_len = int(args.episode_len)
gnn_step = int(args.gnn_step)
rollout_step = int(args.rollout_step)
q_step = int(args.q_step)
n_epoch = int(args.n_epoch)
explore_end_at = float(args.explore_end_at)
anneal_frac = float(args.anneal_frac)
eps = list(np.linspace(float(args.eps0), float(args.eps), int(n_epoch * explore_end_at)))
eps.extend(list(np.linspace(float(args.eps), 0.0, int(n_epoch * anneal_frac))))
eps.extend([0]*int(n_epoch))
save_ckpt_step = int(args.save_ckpt_step)
ddqn = bool(args.ddqn)
num_fold = int(config.fold)
# print(config)
room_acc = []
trip_acc = []

def set_up_logging():
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + 'no_name' + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging = utils.logging(log_path + 'log.txt')
    logging_result = utils.logging_result(log_path + 'record.txt')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_result, log_path

logging, logging_result, log_path = set_up_logging()

def save_fig(dataset, filename):
    '''Saving a figure with scattered points'''
    print_x = []
    print_y = []

    for i in range(len(dataset)//4):
        print_x.append([])
        print_y.append([])

    for index, item in enumerate(dataset):
        # print(index)
        print_x[index//4].append(float(item[0]))
        print_y[index//4].append(float(item[1]))

    colors = cm.rainbow(np.linspace(0, 1, len(print_y)))
    for x, y, c in zip(print_x, print_y, colors):
        plt.scatter(x, y, color=c)
        # plt.plot(x, y, color=c)
    
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig("./result/" + filename + ".png")
    plt.clf()
    plt.close()

def ri_to_label(k, m, room, coordinates):
    result = np.zeros(shape=(k*m, k))
    for i in range(len(coordinates)):
        for j in range(len(room)):
            if i in room[j]:
                result[i][j] = 1
    return result

def save_fig_with_result(k, m, test_episode, x, result_list, n_iter):
    '''
    Writing into a figure with the information of clusters
    '''

    # initializing data
    x_list = []
    y_list = []
    print_x = []
    print_y = []

    for j in range(k * test_episode):
        x_list.append([])
        y_list.append([])
        print_x.append([])
        print_y.append([])

    # x = PCA(n_components=2).fit_transform(x)

    # re-organizing the coordinates to scatter the points,
    # dividing by 4 because we want the points
    # in the same group to have the same color
    for index, item in enumerate(x):
        print_x[index//4].append(float(item[0]))
        print_y[index//4].append(float(item[1]))
    # print("-----")
    # print(print_x)
        

    # This is used for showing which points are predicted to be in the same groups
    for j in range(test_episode):
        for l in range(k*m):
            curr_point = k*m*j+l
            room_number = 0
            if len(result_list[0]) == m:
                for k in range(len(result_list)):
                    if room_number:
                        break
                    else:
                        if curr_point in result_list[k]:
                            room_number = k
            else:
                room_number = result_list[curr_point].index(1.0)
            # print(room_number)
            x_list[room_number+k*j].append(x[m*k*j+l][0])
            y_list[room_number+k*j].append(x[m*k*j+l][1])
    # print(x_list)
    # print("------")
    colors = cm.rainbow(np.linspace(0, 1, len(x_list)))

    # The scatter points represents which points are really in the same groups
    # the plot represents which points are predicted to be in the same groups
    for x, y, x_original, y_original, c in zip(x_list, y_list, print_x, print_y, colors):
        plt.scatter(x_original, y_original, color=c)
        plt.plot(x, y, color=c)

    plt.savefig("./result/RI/sim_output" + str(n_iter) + ".png")
    plt.clf()
    plt.close()

def test_colocation(test_x, test_y, model, fold, split):
    model.eval()
    
    with torch.no_grad():
        if args.model == 'stn':
            out = model(torch.from_numpy(np.array(test_x)).cuda())

        test_triplet = gen_colocation_triplet(test_x, test_y, prevent_same_type = True)
        test_loader = torch.utils.data.DataLoader(test_triplet, batch_size = 1, shuffle = False)
        cnt = 0
        test_anchor_list = []
        for step, batch_x in enumerate(test_loader):
            if args.model == 'stn':
                anchor = batch_x[0].cuda()
                pos = batch_x[1].cuda()
                neg = batch_x[2].cuda()
            # print(anchor.shape)
            # print(len(batch_x))
            output_anchor = model(anchor) 
            output_pos = model(pos) 
            output_neg = model(neg)
            if output_anchor.tolist() not in test_anchor_list:
                test_anchor_list.append(output_anchor.tolist())
            distance_pos = (output_anchor - output_pos).pow(2).sum(1).pow(1/2)
            distance_neg = (output_anchor - output_neg).pow(2).sum(1).pow(1/2)
            if distance_neg > distance_pos:
                cnt += 1

        # f = open("./output/ColocationSensors/test_sensor_" + str(fold) + ".csv", "w+")
        # for i in range(0, len(test_anchor_list)):
        #     writer = csv.writer(f)
        #     writer.writerows(test_anchor_list[i])
        # f.close()

        logging("Testing triplet acc: %f" %(cnt / len(test_triplet)))
    
    trip_acc.append(cnt / len(test_triplet))
    
    """
    """
    test_out = out.cpu()
    # test_out = torch.tensor(PCA(n_components=2).fit_transform(out.cpu()))
    test_corr = np.corrcoef(np.array(test_out))
    # print("----- GA -----")
    # print(test_corr)
    # print("-----")
    # X_transform = PCA().fit_transform(test_out)
    # save_fig(X_transform, str(fold)+"_test")
    # with open("./output/ColocationSensors/test_sensor_" + str(fold) + ".csv", "w+") as f:
    #     csv_writer = csv.writer(f)
    #     for item in test_out.tolist():
    #         csv_writer.writerow(item)

    scio.savemat('./output/corr_' + str(fold) + '.mat', {'corr':test_corr})
    acc_list = []
    best_solution, acc, ground_truth_fitness, best_fitness = run.ga(path_m = './output/corr_' + str(fold) + '.mat', path_c = '10_rooms.json')
    recall, room_wise_acc = utils.cal_room_acc(best_solution)
    
    logging("recall = %f, room_wise_acc = %f:\n" %(recall, room_wise_acc))

    logging("Ground Truth Fitness %f Best Fitness: %f \n" % (ground_truth_fitness, best_fitness))
    logging("Edge-wise accuracy: %f \n" % (acc))

    model.train()
    # model.train()
    room_acc.append(room_wise_acc)
    return best_solution, recall, room_wise_acc

class PreventTypeTripletMiner(BaseTupleMiner):
    def __init__(self, test_index, **kwargs):
        super().__init__(**kwargs)
        self.test_index = test_index

    # def mine(self, embeddings, labels, ref_emb, ref_labels):
    #     a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)
    #     # print(a,p,n)
    #     triplet_mask = torch.logical_and(n%4 != a%4, a != p)
    #     # print(a[triplet_mask], p[triplet_mask], n[triplet_mask])
    #     return a[triplet_mask], p[triplet_mask], n[triplet_mask]

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)
        # print("----- anchor nodes -----")
        
        # exclude_index = np.random.choice(40, 20, replace=False)
        # exclude_index = torch.LongTensor(exclude_index).cuda()
        exclude_index = torch.LongTensor(self.test_index).cuda()
        # These are the indexes to be selected, not excluded
        a_room = torch.div(a, 4, rounding_mode="floor")
        p_room = torch.div(p, 4, rounding_mode="floor")
        n_room = torch.div(n, 4, rounding_mode="floor")
        anc_mask = torch.logical_and(torch.logical_not(torch.isin(p_room, exclude_index)), torch.logical_not(torch.isin(a_room, exclude_index)))
        neg_mask = torch.logical_not(torch.isin(n_room, exclude_index))
        test_mask = torch.logical_and(anc_mask, neg_mask)
        type_mask = torch.logical_and(n%4 != a%4, a != p)
        triplet_mask = torch.logical_and(test_mask, type_mask)
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]

class ToyMiner(BaseTupleMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)
        triplet_mask = torch.logical_and(n%4 != a%4, a != p)
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]

def main():
    G = GraphGenerator(k=k, m=m, ajr=ajr, style=train_graph_style)
    test_indexes = cross_validation_sample(5*k, k)
    # test_indexes = cross_validation_sample(num_fold*k, 20)
    # test_indexes = [[46, 36, 32, 44, 13, 49, 47, 2, 27, 37]]
    # test_indexes = [[13, 39, 19, 26, 44, 23, 36, 11, 15, 28, 12, 38, 24, 22, 10, 16, 18, 8, 43, 3, 20, 49, 1, 47, 34], [45, 9, 33, 40, 14, 46, 37, 2, 25, 4, 27, 6, 48, 17, 41, 29, 0, 35, 42, 31, 21, 7, 32, 5, 30]]
    '''read data & STFT'''
    logging(str(time.asctime( time.localtime(time.time()) )))

    if config.building in ["Soda", "SDH"]:
        print(config.building)
        x, y, true_pos = read_colocation_data(config.building, config.sensor_count, config)
    else:
        x, y, true_pos = read_colocation(config)
    # x, y, true_pos = x[:64], y[:64], true_pos[:64]
    # print("===== x shape =====")
    # print(np.asarray(x).shape)
    x = STFT(x, config)
    # print(np.asarray(x).shape)
    logging("%d total sensors, %d frequency coefficients, %d windows\n" % (len(x), x[0].shape[0], x[0].shape[1]))

    print("test indexes:\n", test_indexes)

    fold_recall = []
    fold_room_acc = []
    
    """ 5-fold cross validation"""
    # 5 folds
    # 40 as training, 10 testing
    for fold, test_index in enumerate(test_indexes):
        logging("Now training fold: %d" %(fold))
        # split training & testing
        print("Test indexes: ", test_index)
        train_x, train_y, test_x, test_y = split_colocation_train(x, y, test_index, args.split)
        train_x_reserved = np.stack(x, axis=0)
        print(train_x_reserved.shape)
        test_x_reserved = np.stack(test_x, axis=0)
        print(test_x_reserved.shape)
        # train_x_reserved = np.concatenate((train_x_reserved, test_x_reserved), axis=0)
        # print(train_x_reserved.shape)

        train_x = gen_colocation_triplet(train_x, train_y, prevent_same_type=True)
        total_triplets = len(train_x)
        logging("Total training triplets: %d\n" % (total_triplets))
        
        if args.loss == 'triplet':
            criterion = tripletLoss(margin = 1).cuda()
        elif args.loss == 'comb':
            criterion = combLoss(margin = 1).cuda()

        if args.model == 'stn':
            model = STN(config.dropout, 2 * config.k_coefficient).cuda()
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(pytorch_total_params)
            
            # model1 = STN(config.dropout, 2 * config.k_coefficient).cuda()
            # model2 = STN(config.dropout, 2 * config.k_coefficient).cuda()
            # model3 = STN(config.dropout, 2 * config.k_coefficient).cuda()
            # model4 = STN(config.dropout, 2 * config.k_coefficient).cuda()
            
        if config.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, momentum = 0.9, weight_decay = config.weight_decay)
        elif config.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

        # if config.optim == 'SGD':
        #     optimizer = torch.optim.SGD(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()) +list(model4.parameters()), lr = config.learning_rate, momentum = 0.9, weight_decay = config.weight_decay)
        # elif config.optim == 'Adam':
        #     optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()) +list(model4.parameters()), lr = config.learning_rate, weight_decay = config.weight_decay)

        if config.grad_norm > 0:
            nn.utils.clip_grad_value_(model.parameters(), config.grad_norm)
            for p in model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -config.grad_norm, config.grad_norm))

        # print("Model : ", model)
        print("Criterion : ", criterion)

        # print(torch.cuda.device_count())
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)

        train_loader = torch.utils.data.DataLoader(train_x, batch_size = config.batch_size, shuffle = True, num_workers=torch.cuda.device_count()//2)

        '''Deep Metric Learning'''
        # try:
        #     model.load_state_dict(torch.load("./output/DML_model_" + str(fold) + "_" + str(args.seed))) # DML
        #     for param in model.parameters():
        #         param.requires_grad = False
        # except FileNotFoundError:

        gif_fig_list = []
        gif_fig_list_train = []
        T1 = time.time()

        # distance = LpDistance(power=2)
        # # distance = CosineSimilarity()
        # ang_loss = AngularLoss()
        # trip_loss = TripletMarginLoss(margin=1.0, distance=SNRDistance(), triplets_per_anchor=1) # 480 per anchor
        # mining_func = PreventTypeTripletMiner(test_index)
        # criterion = MultipleLosses([ang_loss, trip_loss], miners=[mining_func, mining_func])
        # # mining_func = BatchHardMiner()
        # # criterion = MultipleLosses([trip_loss])
        # # criterion = MultipleLosses([ang_loss, trip_loss])
        # # criterion = trip_loss
        # perturbation_shape = train_x_reserved.shape
        # test_x_vali = torch.Tensor(np.stack(x[160:], axis=0)).cuda()
        # # anc-pos ~1.0
        # # anc-neg ~1.4
        train_losses = []
        test_losses = []
        
        for epoch in range(config.epoch):
            # if epoch % 10 == 0:
            #     train_x_perturbed = my_augmenter.augment(np.swapaxes(train_x_reserved, 1, 2))
            #     train_x_perturbed = torch.Tensor(np.swapaxes(train_x_perturbed, 1, 2)).cuda()
            #     perturbation = torch.normal(mean=0, std=0.3, size=perturbation_shape).cuda()
            #     amplification = torch.normal(mean=1, std=0.1, size=perturbation_shape).cuda()

            # if epoch == 980:
            #     mining_func = ToyMiner()
            #     criterion = MultipleLosses([ang_loss, trip_loss], miners=[mining_func, mining_func])

            if epoch % 100 == 0:
                logging("Now training %d epoch ......\n" % (epoch + 1))
            total_triplet_correct = 0
            # train_x_perturbed = amplification*torch.Tensor(train_x_reserved).cuda()+perturbation
            # train_x_perturbed = torch.Tensor(train_x_reserved).cuda()
            
            # # train_x_perturbed = train_x_perturbed.reshape(ajr+1, m, 2*config.k_coefficient, -1)
            # # embedding_1 = model1(train_x_perturbed[:,0,:,:])
            # # embedding_2 = model2(train_x_perturbed[:,1,:,:])
            # # embedding_3 = model3(train_x_perturbed[:,2,:,:])
            # # embedding_4 = model4(train_x_perturbed[:,3,:,:])
            # # embedding = torch.stack([embedding_1, embedding_2, embedding_3, embedding_4], dim=1).reshape(m*(ajr+1),-1)
            
            # embedding = model(train_x_perturbed)
            # # mining_func.mine(embedding, torch.LongTensor(train_y), embedding, torch.LongTensor(train_y)) # the smaller, the less triplets selected, the harder the problem is
            # # embedding = model(torch.Tensor(train_x_reserved).cuda())
            
            # # indices_tuple = mining_func(embedding, torch.Tensor(train_y))
            # # print(embedding.shape)
            # # print(torch.LongTensor(train_y))
            # # print(torch.cat([torch.LongTensor(train_y), torch.full([40], 100)]).shape)
            # # loss = criterion(embedding, torch.cat([torch.LongTensor(train_y), torch.full([40], 100)]))
            # # loss = criterion(embedding, torch.cat([torch.Tensor(train_y), torch.Tensor(test_y)]))
            # loss = criterion(embedding, torch.Tensor(y))
            # # loss = criterion(embedding, torch.LongTensor(train_y))
            
            # # print(loss) 
            # train_losses.append(loss.detach().cpu().tolist())
            # optimizer.zero_grad()           
            # loss.backward()               
            # optimizer.step()

            # model.eval()
            # test_embedding = model(torch.Tensor(test_x_reserved).cuda())
            # test_loss = criterion(test_embedding, torch.LongTensor(test_y))
            # test_losses.append(test_loss.detach().cpu().tolist())
            # model.train()

            # Disabled to test trip loss provided by package
            for step, batch_x in enumerate(train_loader):
                # print(step)
                # optimize time?
                if args.model == 'stn':
                    anchor = batch_x[0].cuda()
                    pos = batch_x[1].cuda()
                    neg = batch_x[2].cuda()

                output_anchor = model(anchor) 
                output_pos = model(pos) 
                output_neg = model(neg)
                # T3 = time.time()

                loss, triplet_correct = criterion(output_anchor, output_pos, output_neg)
                total_triplet_correct += triplet_correct.item()

                # T4 = time.time()

                optimizer.zero_grad()           
                loss.backward()               
                optimizer.step()

                # T5 = time.time()
                # print("backward time: ")
                # print(T5-T4)

                if step % 200 == 0 and step != 0:
                    logging("loss " + str(loss) + "\n")
                    logging("triplet_acc " + str(triplet_correct.item()/config.batch_size) + "\n")
                    # solution, recall, room_wise_acc = test_colocation(test_x, test_y, model, fold, args.split)
                    trip_acc.append(0.0)
                    train_losses.append(triplet_correct.item()/config.batch_size)
                    test_losses.append(trip_acc[-1])
                    
                    model.eval()
                    test_out = model(torch.from_numpy(test_x_reserved).cuda()).cpu()
                    train_out = model(torch.from_numpy(train_x_reserved[:80].reshape(-1, 64, 998)).cuda()).cpu()
                    X_test_transform = PCA(random_state=42).fit_transform(test_out.detach().cpu())
                    save_fig(X_test_transform, str(fold+1) + "_" + str(step+1) + "test_pretrain")
                    X_train_transform = PCA(random_state=42).fit_transform(train_out.detach().cpu())
                    save_fig(X_train_transform, str(fold+1) + "_" + str(step+1) + "train_pretrain")

                    del test_out
                    torch.cuda.empty_cache()
                    model.train()

            solution, recall, room_wise_acc = test_colocation(test_x, test_y, model, fold, args.split)

            logging("Triplet accuracy: %f" % (total_triplet_correct/total_triplets))
            # solution = solution.tolist()

            # logging_result("fold: %d, epoch: %d\n" % (fold, epoch))
            # logging_result("Acc: %f\n" %(recall))
            # logging("fold: %d, epoch: %d\n" % (fold, epoch))
            # logging("Acc: %f\n" %(recall))
            
        #     if epoch == config.epoch-1:
        #     # if epoch % 1000 == 0 or epoch == config.epoch-1:
        #         solution, recall, room_wise_acc = test_colocation(test_x_reserved, test_y, model, fold, args.split)
        #         # solution, recall, room_wise_acc = test_colocation(train_x_reserved[-40:], train_y[-40:], model, fold, args.split)
                    
            # if epoch % 25 == 0:
            #     X_train_transform = X_transform[:-40]
            #     X_test_transform = X_transform[-40:]
            '''Saving a figure with scattered points'''
        #       print_x = []
        #       print_y = []

        #         for i in range(len(X_train_transform)//4):
        #             print_x.append([])
        #             print_y.append([])

        #         for index, item in enumerate(X_train_transform):
        #             # print(index)
        #             print_x[index//4].append(float(item[0]))
        #             print_y[index//4].append(float(item[1]))

        #         colors = cm.rainbow(np.linspace(0, 1, len(print_y)))
        #         for a, b, c in zip(print_x, print_y, colors):
        #             plt.scatter(a, b, color=c, marker="o")
        #             # plt.plot(x, y, color=c)

        #         print_x = []
        #         print_y = []

        #         for i in range(len(X_val_transform)//4):
        #             print_x.append([])
        #             print_y.append([])

        #         for index, item in enumerate(X_val_transform):
        #             # print(index)
        #             print_x[index//4].append(float(item[0]))
        #             print_y[index//4].append(float(item[1]))

        #         colors = cm.rainbow(np.linspace(0, 1, len(print_y)))
        #         for a, b, c in zip(print_x, print_y, colors):
        #             plt.scatter(a, b, color=c, marker="D")
                
        #         plt.xlim(-1,1)
        #         plt.ylim(-1,1)
        #         plt.savefig("./result/" + str(fold+1) + "_" + str(epoch+1) + "train_and_test" + ".png")
        #         plt.clf()
        #         plt.close()

        #         # save_fig(X_transform, str(fold+1) + "_" + str(epoch+1) + "train_pretrain_10rooms")
        #         gif_fig_list_train.append("./result/" + str(fold+1) + "_" + str(epoch+1) + "train_and_test" + ".png")
        #         model.train()


        # T2 = time.time()
        # print(T2-T1)

        # iter_axis = np.arange(config.epoch)
        iter_axis = np.arange(len(test_losses))
        plt.plot(iter_axis, train_losses, label='train')
        plt.plot(iter_axis, test_losses, label='test')
        plt.ylim(0.5,1)
        plt.savefig("./result/test_v_train" + str(int(time.time())) + ".png") 
        plt.clf()
        plt.close()

        # # ims = [imageio.imread(f) for f in gif_fig_list]
        # # imageio.mimwrite("./result/test" + str(fold) +".gif", ims, fps=5)

        # ims_train = [imageio.imread(f) for f in gif_fig_list_train]
        # imageio.mimwrite("./result/train" + str(fold) +".gif", ims_train, fps=5)
        # # torch.save(model.state_dict(), "./output/DML_model_" + str(fold) + "_" + str(args.seed))

        RI_train_in = torch.Tensor(train_x_reserved).cuda()
        RI_test_in = torch.Tensor(np.stack(test_x, axis=0)).cuda()
        # RI_test_in = test_x
        # print(train_x_reserved.shape)
        # print(RI_train_in.shape)
        # print(np.stack(test_x, axis=0).shape)
        # print(RI_test_in.shape)
        # break
        # for param in model.parameters():
        #     param.requires_grad = False
        alg = DQN(graph_generator=G, hidden_dim=h, action_type=action_type
                        , gamma=gamma, eps=.1, lr=lr, action_dropout=action_dropout
                        , sample_batch_episode=sample_batch_episode
                        , replay_buffer_max_size=replay_buffer_size
                        , epi_len=episode_len
                        , new_epi_batch_size=n_episode
                        , cuda_flag=cuda_flag
                        , explore_method=explore_method
                        , priority_sampling=priority_sampling
                        , DML_model=model
                        , model_file="./result/save_model/dqn_197000_1")

        model_version = fold

        if save_log:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = './log/' + current_time + '/' + 'fold' + str(fold) + '_pipeline'
            train_summary_writer = SummaryWriter(train_log_dir)
            accuracy_dir = './result/accuracy/RealTrainingRealTesting/' + train_log_dir + '/'
            if not os.path.exists(accuracy_dir):
                os.makedirs(accuracy_dir)
            accuracy_file = accuracy_dir + 'accuracy.txt'
            with open(accuracy_file, 'w+') as the_file:
                the_file.write("Learning Rate: " + str(lr) + "\n")
                the_file.write("Action Dropout: " + str(action_dropout) + "\n")
                the_file.write("Replay Buffer Size: " + str(replay_buffer_size) + "\n")
                the_file.write("Number of Episodes: " + str(n_episode) + "\n")
                the_file.write("Exploration rate: " + str(explore_end_at) + "\n")
                the_file.write("---------- Now Training Fold:" + str(fold) + " ----------\n")

        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_log_dir = './log/' + current_time + '/' + 'fold' + str(fold) + '_pipeline'
        # train_summary_writer = SummaryWriter(train_log_dir)
        # accuracy_dir = './result/accuracy/RealTrainingRealTesting/' + train_log_dir + '/'
        
        # if not os.path.exists(accuracy_dir):
        #     os.makedirs(accuracy_dir)
        # with open(accuracy_dir + 'param.txt', 'w+') as the_file:
        #     the_file.write("Model Params: " + "\n")
        for param in model.parameters():
            param.requires_grad = True
        # global perturbation
        for n_iter in tqdm(range(n_epoch)):
            model.train()

            for param in model.parameters():
                param.requires_grad = True

            # if n_iter % 10 == 0:
            #     amplification =  torch.normal(mean=1, std=0.1, size=train_x_reserved.shape).cuda()
            #     perturbation = torch.normal(mean=0, std=0.3, size=train_x_reserved.shape).cuda()
                
            # alg.eps = eps[n_iter]
            # print((amplification*RI_train_in+perturbation).shape)
            
            RI_train_out = model(RI_train_in) # input -> DML -> output1
            RI_train_out = RI_train_out.cpu()
            # RL_train = torch.reshape(RI_train_out, (-1, m, 805))

            '''Reinforcement Learning'''
            # Prepare the graph to be trained
            
            # with open(accuracy_dir + 'param.txt', 'a+') as the_file:
            #     for name, param in alg.DML_model.named_parameters():
            #         print(param.requires_grad)
            #         the_file.write(str(n_iter) + ": " + str(name) + str(param.data))

            # if n_iter >= n_epoch//2 and n_iter % validation_step == 0:
            # if n_iter % validation_step == 0:
            #     # if n_iter % 3*validation_step == 0:
            #     for param in model.parameters():
            #         # param.requires_grad = not param.requires_grad
            #         param.requires_grad = True
            #     # else:
            #     #     for param in alg.DML_model.parameters():
            #     #         param.requires_grad = False
            #     # print(RI_train_out)

            #     # test_corr = 1-np.corrcoef(np.array(RI_train_out.reshape((-1,805)).detach()[:80]))
            #     # X_transform = MDS(dissimilarity='precomputed').fit_transform(test_corr)
            #     X_transform = PCA().fit_transform(np.array(RI_train_out.reshape((-1,805)).detach()[0:80]))
            #     save_fig(X_transform, str(n_iter+20000))

            #     # test_corr = 1-np.corrcoef(np.array(RI_train_out.reshape((-1,805)).detach()[80:160]))
            #     # X_transform = MDS(dissimilarity='precomputed').fit_transform(test_corr)
            #     X_transform = PCA().fit_transform(np.array(RI_train_out.reshape((-1,805)).detach()[80:160]))
            #     save_fig(X_transform, str(n_iter+20001))

            #     # test_corr = 1-np.corrcoef(np.array(RI_train_out.reshape((-1,805)).detach()))
            #     # X_transform = MDS(dissimilarity='precomputed').fit_transform(test_corr)
            #     X_transform = PCA().fit_transform(np.array(RI_train_out.reshape((-1,805)).detach()))
            #     save_fig(X_transform, str(n_iter+20002))

            if n_iter % 100 == 0:
                model.eval()
                # print(RI_test_in.shape)
                # print(model(RI_test_in).shape)
                RI_test_out_visualization = torch.reshape(model(RI_test_in), (-1, 176))
                # print(RI_test_out_visualization.shape)
                X_transform = PCA(svd_solver="arpack").fit_transform(np.array(RI_test_out_visualization.detach().cpu()))
                save_fig(X_transform, str(n_iter+20000)+"_test_" + str(fold))
                gif_fig_list.append("./result/" + str(20000+n_iter) + "_test_" + str(fold)+ ".png")
                RI_test_out_visualization = RI_test_out_visualization.detach().cpu()
                del RI_test_out_visualization
                torch.cuda.empty_cache()

                RI_train_out_visualization = model(RI_train_in)
                X_transform = PCA(svd_solver="arpack").fit_transform(np.array(RI_train_out_visualization.detach().cpu()[40:80]))
                save_fig(X_transform, str(n_iter+20000)+"_train_" + str(fold))
                gif_fig_list_train.append("./result/" + str(20000+n_iter) + "_train_" + str(fold) + ".png")
                del RI_train_out_visualization
                torch.cuda.empty_cache()
                model.train()
            # print(id(model))
            # print(id(alg.DML_model))

            # RI_test_out = torch.reshape(alg.DML_model(RI_test_in), (-1, 805))
        
            # test_corr = 1-np.corrcoef(np.array(RI_test_out.detach().cpu()[:40]))
            # X_transform = MDS(dissimilarity='precomputed').fit_transform(test_corr)
            # # X_transform = PCA().fit_transform(RI_test_out.detach().cpu().numpy())
            # save_fig(X_transform, str(n_iter)+"_test")
            # gif_fig_list.append("./result/" + str(n_iter) + "_test" + ".png")

            weights = torch.ones(k*m)
            index = torch.cat([torch.multinomial(weights, k, replacement=False) for _ in range(n_episode)], 0)
            
            train_g = G.generate_graph(x=RI_train_out.detach(), index=index.reshape(-1, k), batch_size=n_episode, cuda_flag=cuda_flag)

            del RI_train_out
            torch.cuda.empty_cache()
            
            # Issue: train_g's deep copy cannot be created. 
            # Solved by regenerating a graph using the same embedding
            log = alg.train_dqn(
                            target_bg=train_g
                            , epoch=n_iter
                            , batch_size=batch_size
                            , num_episodes=n_episode
                            , episode_len=episode_len
                            , gnn_step=gnn_step
                            , q_step=q_step
                            , grad_accum=grad_accum
                            , rollout_step=rollout_step
                            , ddqn=ddqn
                            , raw_x=amplification*RI_train_in+perturbation) #amplification*RI_train_in+perturbation

            

            # print(torch.autograd.grad(Q.sum(), train_subsample)[0])

            if n_iter % target_update_step == target_update_step - 1:
                alg.update_target_net()

            if n_iter % save_ckpt_step == save_ckpt_step - 1:
                with open('./output/joint_training/dqn_'+ str(n_iter + 1), 'wb') as model_file:
                    pickle.dump(alg.model, model_file)
                with open('./output/joint_training/dml_'+ str(n_iter + 1), 'wb') as model_file:
                    pickle.dump(model, model_file)

            print('Epoch: {}. R: {}. Q error: {}. H: {}. Triplet Error: {}'
                .format(n_iter
                , np.round(log.get_current('tot_return'), 2)
                , np.round(log.get_current('Q_error'), 3)
                , np.round(log.get_current('entropy'), 3)
                # , 0
                , np.round(log.get_current('Triplet_error'), 3)
                ))

            if save_log:
                train_summary_writer.add_scalar('Q_error', log.get_current('Q_error'), n_iter)
                train_summary_writer.add_scalar('S', log.get_current('tot_return'), n_iter)
                train_summary_writer.add_scalar('Trip_error', log.get_current('Triplet_error'), n_iter)

            # if n_iter % validation_step == 1 or n_iter % validation_step == 9:
            #     X_transform = PCA().fit_transform(RI_train_out.reshape((-1,805))[:40].cpu().detach())
            #     save_fig(X_transform, str(n_iter))

            # if n_iter >= (n_epoch/2) and n_iter%(validation_step) == 0:
            #     for param in model.parameters():
            #         param.requires_grad = not param.requires_grad

            if validation_step and n_iter % validation_step == 0:
                # for name, param in model.named_parameters():
                #     print(name, param.data)

                solution, recall, room_wise_acc = test_colocation(test_x, test_y, model, fold, args.split)

                model.eval()
                RI_test_out = torch.reshape(model(RI_test_in), (-1, 176))
                # with open(accuracy_dir + 'param.txt', 'a+') as the_file:
                #     the_file.write(str(n_iter) + ": \n" + str(torch.mean(RI_test_out)) + "\n")

                test_corr = 1-np.corrcoef(np.array(RI_test_out.detach().cpu()))
                # np.set_printoptions(threshold=sys.maxsize)
                # print(test_corr)
                X_transform = MDS(dissimilarity='precomputed').fit_transform(test_corr)
                # X_transform = PCA().fit_transform(np.array(RI_test_out.detach().cpu()))
                save_fig_with_result(k, m, 1, X_transform, solution, str(n_iter) + "GA_output")

                avr_accuracy = []
                avr_initial_value = []
                final_value = []
                gt_cluster_assignment = torch.nn.functional.one_hot(torch.arange(k).repeat_interleave(m).reshape(-1,1)).squeeze(1)
                # print(gt_cluster_assignment)
                gt_test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag, label_input=gt_cluster_assignment)
                print("Cut of groundTruth assignment: " + str(gt_test_g.kcut_value))
                # print(solution)
                # print(torch.div(torch.Tensor(solution), 4, rounding_mode="floor").reshape(-1,1).to(torch.int64))

                GA_cluster_assignment = torch.nn.functional.one_hot(torch.div(torch.Tensor(solution), 4, rounding_mode="floor").reshape(-1,1).to(torch.int64))
                # print(GA_cluster_assignment)
                GA_test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag, label_input=GA_cluster_assignment)
                print("Cut of GA assignment: " + str(GA_test_g.kcut_value))

                for _ in range(10):
                    test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag)
                    
                    test = test_summary(alg=alg, graph_generator=G, action_type=action_type, q_net=readout, forbid_revisit=0)
                    test.run_test(problem=test_g, trial_num=1, batch_size=test_episode, action_dropout=action_dropout, gnn_step=gnn_step,
                                episode_len=episode_len, explore_prob=0.0, Temperature=1e-8, cuda_flag=cuda_flag, save_log=True, k=k, m=m)
                    test.show_result(k, m)

                    # saving the result as figure
                    
                    # saving figures with scatter points and connected edges
                    # pca = PCA(n_components=2)
                    # X = PCA().fit_transform([RI_test_out].cpu().detach())
                    # save_fig(X, "test_graph_visualization" + str(n_iter))
                    # print(torch.argmax(test.bg.ndata['label'], dim=1).reshape(k, m))
                    # print(torch.argmax(test.bg.ndata['label'], dim=1).reshape(k, m).numpy()) 
                    
                    RL_solution = 4*torch.argmax(test.bg.ndata['label'], dim=1).reshape(1, k, m) + torch.arange(m).repeat(k).reshape(1,k,m).cuda()
                    RL_solution = RL_solution.detach().cpu().numpy()
                    print(RL_solution)
                    corr_func = corr_score.compile_solution_func(1-test_corr, m)
                    fitnesses: np.ndarray = ga.fitness(RL_solution, corr_func)
                    print(fitnesses)
                    print(test.acc)
                    save_fig_with_result(k, m, 1, X_transform, test.bg.ndata['label'].tolist()[:40], str(fold) + str(n_iter) + "RL_output" + str(_))

                    # calculate the accuracy
                    avr_accuracy.append(test.acc)
                    # print(test.bg)
                    avr_initial_value.append(torch.mean(test.S).item())
                    final_value.append(np.float64(np.mean(test.bg.kcut_value.numpy())))

                best_index = final_value.index(min(final_value))

                # writing to a file and the tensorboard
                if save_log:
                    with open(accuracy_file, 'a') as the_file:
                        the_file.write('Epoch: {}. Q error: {}. Best Accuracy: {} %. Best Final Value: {}. Mean Accuracy: {}. Std of Accuracy: {}. Mean Final Value: {}. Std of Final Value: {}. GA Accuracy: {}'
                    .format(n_iter
                    , np.round(log.get_current('Q_error'), 3) # q_error
                    , np.round(avr_accuracy[best_index], 2) # best accuracy
                    , np.round(final_value[best_index], 2) # best final value
                    # , np.round(avr_initial_value[best_index], 2) # initial value corresponding to the best trial
                    , np.round(statistics.mean(avr_accuracy), 2) # mean accuracy
                    , np.round(statistics.stdev(avr_accuracy), 2) # std of accuracy
                    , np.round(statistics.mean(final_value), 2) # mean final value
                    , np.round(statistics.stdev(final_value), 2) # std of final value
                    , np.round(room_wise_acc, 2) # std of final value
                    ) + "\n")
                
                else:
                    print(avr_accuracy)

                # choice 1
                # RI_test_in = RI_test_in.cpu().detach()
                # test_x = test_x.cpu().detach()
                # del RI_test_in
                # del test_x

                model.train()

                RI_test_out = RI_test_out.cpu().detach()
                del RI_test_out
                del test_g
                del test

            # Causes of memory leak
            # train_subsample = train_subsample.cpu().detach()
            # RL_train = RL_train.cpu().detach()

            # RI_train_out = RI_train_out.detach()
            # del RI_train_out
            for node_attr in train_g.ndata.keys():
                if node_attr in ('adj', 'label'):
                    train_g.ndata[node_attr] = train_g.ndata[node_attr].detach().cpu()
            del train_g
            del log
            # del train_subsample
            # del RL_train

            # disables if choice 1
            # RI_train_out = RI_train_out.cpu().detach()
            # del RI_train_out

            # choice 1
            # embedding = embedding.cpu().detach()
            # del embedding

            # ???
            # train_x_reserved = train_x_reserved.cpu().detach()
            # del train_x_reserved

            torch.cuda.empty_cache()

        # write_data = model(torch.from_numpy(np.array(RL_train_x)).cuda()).tolist()
        # # print(len(write_data))
        # # print(len(write_data[0]))
        # f = open("./output/ColocationSensors/sensor_" + str(fold) + ".csv", "w+")
        # writer = csv.writer(f)
        # for i in range(len(write_data)):
        #     writer.writerows([write_data[i]])
        # f.close()


        RI_train_in = RI_train_in.cpu().detach()
        RI_test_in = RI_test_in.cpu().detach()
        # if save_log:
        #     x_torch = torch.Tensor(x).cuda()
        #     embedding = model(x_torch)
        #     # print(embedding)
        #     df = pd.DataFrame(embedding.cpu().detach().numpy()) # convert to a dataframe
        #     # print(df)
        #     df.to_csv("./output/ColocationSensors/co_trained_sensor_" + str(fold) + ".csv", header=None, index=False) #save to file
        #     x_torch = x_torch.cpu().detach()
        #     embedding = embedding.cpu().detach()
        #     del x_torch, embedding
        model = model.cpu()
        criterion = criterion.cpu()
        del model, criterion
        del RI_train_in, RI_test_in
        torch.cuda.empty_cache()

        # ims = [imageio.imread(f) for f in gif_fig_list]
        # imageio.mimwrite("./result/test" + str(fold) +".gif", ims, fps=5)

        # ims_train = [imageio.imread(f) for f in gif_fig_list_train]
        # imageio.mimwrite("./result/train" + str(fold) +".gif", ims_train, fps=5)
    # logging("Final recall : %f \n" % (np.array(fold_recall).mean()))
    # logging("Final room accuracy : %f \n" % (np.array(fold_room_acc).mean()))
    print(room_acc)
    print(trip_acc)

if __name__ == '__main__':
    main()
    