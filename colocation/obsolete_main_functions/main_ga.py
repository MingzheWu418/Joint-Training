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

import itertools

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
print("-----")
print(torch.__version__)
print(torch.version.cuda)

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
    parser.add_argument('-loss', default='comb', type=str,
                        choices=['triplet', 'comb'])
    parser.add_argument('-seed', default=765, type=int,
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
    parser.add_argument('--m', default='3', help="cluster size")
    parser.add_argument('--ajr', default=29, help="")
    parser.add_argument('--h', default=128, help="hidden dimension")
    parser.add_argument('--rollout_step', default=1)
    parser.add_argument('--q_step', default=2)
    parser.add_argument('--batch_size', default=100, help='')
    parser.add_argument('--n_episode', default=10, help='')
    parser.add_argument('--episode_len', default=100, help='')
    parser.add_argument('--grad_accum', default=1, help='')
    parser.add_argument('--action_type', default='swap', help="")
    parser.add_argument('--gnn_step', default=4, help='')
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
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--action_dropout', type=float, default=1.0)
    parser.add_argument('--n_epoch', default=0)
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
# trip_acc = []

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

    for i in range(len(dataset)//m):
        print_x.append([])
        print_y.append([])

    for index, item in enumerate(dataset):
        # print(index)
        print_x[index//m].append(float(item[0]))
        print_y[index//m].append(float(item[1]))

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
        print_x[index//m].append(float(item[0]))
        print_y[index//m].append(float(item[1]))
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

def run_ga(x, model, fold):   
    model.eval()
    try:
        out = model(torch.from_numpy(np.array(x)).cuda())
    except:
        out = model(x)
    test_out = out.detach().cpu()
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
    best_solution, acc, ground_truth_fitness, best_fitness = run.ga(path_m = './output/corr_' + str(fold) + '.mat', path_c = '10_rooms_copy.json')
    recall, room_wise_acc = utils.cal_room_acc(best_solution)
    
    logging("recall = %f, room_wise_acc = %f:\n" %(recall, room_wise_acc))

    logging("Ground Truth Fitness %f Best Fitness: %f \n" % (ground_truth_fitness, best_fitness))
    logging("Edge-wise accuracy: %f \n" % (acc))

    model.train()
    return best_solution, recall, room_wise_acc

def test_colocation(test_x, test_y, model, fold, split):
    model.eval()
    trip_acc = []
    with torch.no_grad():
        if args.model == 'stn':
            try:
                out = model(torch.from_numpy(np.array(test_x)).cuda())
            except:
                out = model(test_x)
        # print(test_y)
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
    test_out = out.detach().cpu()
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
    return best_solution, recall, room_wise_acc, trip_acc

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
        a_room = torch.div(a, m, rounding_mode="floor")
        p_room = torch.div(p, m, rounding_mode="floor")
        n_room = torch.div(n, m, rounding_mode="floor")
        anc_mask = torch.logical_and(torch.logical_not(torch.isin(p_room, exclude_index)), torch.logical_not(torch.isin(a_room, exclude_index)))
        neg_mask = torch.logical_not(torch.isin(n_room, exclude_index))
        test_mask = torch.logical_and(anc_mask, neg_mask)
        type_mask = torch.logical_and(n%m != a%m, a != p)
        triplet_mask = torch.logical_and(test_mask, type_mask)
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]

class ToyMiner(BaseTupleMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)

        triplet_mask = torch.logical_and(n%m != a%m, a != p)
        # random_mask = torch.randint(0,2, (triplet_mask.shape)).bool()
        # triplet_mask = torch.logical_and(triplet_mask, random_mask.cuda())
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]

class GAMiner(BaseTupleMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        # print("------")
        # print(labels, ref_labels)
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)

        # print("-----")
        # print(a,p,n)
        triplet_mask = torch.logical_and(n%m != a%m, a != p)

        # print(a[triplet_mask])
        # print(p[triplet_mask])
        # print(n[triplet_mask])
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]

def main():
    G = GraphGenerator(k=k, m=m, ajr=ajr, style=train_graph_style)
    # test_indexes = cross_validation_sample(50, k)
    test_indexes = cross_validation_sample(30, k)
    # test_indexes =  [[38, 10, 24, 35, 0], [45, 26, 9, 29, 16], [46, 36, 32, 44, 13], [25, 23, 19, 11, 4], [31, 21, 12, 3, 39], [43, 18, 33, 48, 41], [30, 28, 20, 22, 42], [49, 47, 2, 27, 37], [5, 34, 6, 8, 14], [15, 17, 1, 7, 40]]
    '''read data & STFT'''
    logging(str(time.asctime( time.localtime(time.time()) )))

    # print(config.building)
    if config.building in ["Soda", "SDH"]:
        print(config.building)
        x, y, true_pos = read_colocation_data(config.building, m, config)
    else:
        x, y, true_pos = read_colocation(config)
    # x, y, true_pos = x[:64], y[:64], true_pos[:64]
    # print("===== x shape =====")
    # print(np.asarray(x).shape)
    x = STFT(x, config)
    # print(np.asarray(x).shape)
    # print(y)
    logging("%d total sensors, %d frequency coefficients, %d windows\n" % (len(x), x[0].shape[0], x[0].shape[1]))

    print("test indexes:\n", test_indexes)

    fold_recall = []
    fold_room_acc = []

    triplet_accuracy = []
    room_accuracy = []
    mean_room_accuracy = []

    """ 5-fold cross validation"""
    # 5 folds
    # 40 as training, 10 testing
    for fold, test_index in enumerate(test_indexes):
        
        triplet_accuracy.append([])
        room_accuracy.append([])
        mean_room_accuracy.append([])
        # print("----- Each Fold -----")
        # nvmlInit()
        # gpu_mem_monitor = nvmlDeviceGetHandleByIndex(1)
        # info = nvmlDeviceGetMemoryInfo(gpu_mem_monitor)
        # print(f'total    : {info.total}')
        # print(f'free     : {info.free}')
        # print(f'used     : {info.used}')

        logging("Now training fold: %d" %(fold))
        # split training & testing
        print("Test indexes: ", test_index)
        train_x, train_y, test_x, test_y = split_colocation_train(x, y, test_index, args.split)
        # print(train_y)
        # print("------")
        train_x_reserved = np.stack(train_x, axis=0)
        print(train_x_reserved.shape)
        test_x_reserved = np.stack(test_x, axis=0)
        print(test_x_reserved.shape)

        # train_x = gen_colocation_triplet(train_x, train_y)
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
        train_x = torch.Tensor(np.array(train_x)).reshape(-1, m, 64, 6491)
        train_y = torch.IntTensor(np.array(train_y)).reshape(-1, m)
        # print(train_y)
        # print(train_y.shape)
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, drop_last=False)

        '''Deep Metric Learning'''
        # try:
        #     model.load_state_dict(torch.load("./output/DML_model_" + str(fold) + "_" + str(args.seed))) # DML
        #     for param in model.parameters():
        #         param.requires_grad = False
        # except FileNotFoundError:

        gif_fig_list = []
        gif_fig_list_train = []

        # distance = LpDistance(power=2)
        distance = CosineSimilarity()
        ang_loss = AngularLoss()
        trip_loss = TripletMarginLoss(margin=1.0, distance=SNRDistance(), triplets_per_anchor=100) # 480 per anchor
        mining_func = ToyMiner()
        # mining_func = PreventSameType(test_index)
        criterion = MultipleLosses([ang_loss, trip_loss], miners=[mining_func, mining_func])
        # mining_func = BatchHardMiner()
        # criterion = MultipleLosses([trip_loss], miners=[mining_func])
        # criterion = MultipleLosses([ang_loss, trip_loss])
        # criterion = trip_loss
        # test_x_vali = torch.Tensor(np.stack(x[160:], axis=0)).cuda()
        train_losses = []
        test_losses = []
        
        for epoch in range(config.epoch):

            # print("----- Each Epoch -----")
            # nvmlInit()
            # gpu_mem_monitor = nvmlDeviceGetHandleByIndex(1)
            # info = nvmlDeviceGetMemoryInfo(gpu_mem_monitor)
            # print(f'total    : {info.total}')
            # print(f'free     : {info.free}')
            # print(f'used     : {info.used}')


            # if epoch == 980:
            #     mining_func = ToyMiner()
            #     criterion = MultipleLosses([ang_loss, trip_loss], miners=[mining_func, mining_func])

            if epoch % 100 == 0:
                logging("Now training %d epoch ......\n" % (epoch + 1))
            total_triplet_correct = 0

            for step, (batch_x, batch_y) in enumerate(train_loader):
                
                # print(batch_x.shape)
                # print(batch_y)

                # perturbation_shape = batch_x.reshape(-1, 64, 6496).shape
                # if epoch % 10 == 0:
                # perturbation = torch.normal(mean=0, std=0.3, size=perturbation_shape).cuda()
                # amplification = torch.normal(mean=1, std=0.1, size=perturbation_shape).cuda()

                # print("----- Each Batch -----")
                # nvmlInit()
                # gpu_mem_monitor = nvmlDeviceGetHandleByIndex(1)
                # info = nvmlDeviceGetMemoryInfo(gpu_mem_monitor)
                # print(f'total    : {info.total}')
                # print(f'free     : {info.free}')
                # print(f'used     : {info.used}')

                # print(batch_x.shape)
                # print(perturbation.shape)
                # print(batch_y)
                # train_x_perturbed = amplification*batch_x.reshape(-1, 64, 6496).cuda()+perturbation
                train_x_perturbed = batch_x.reshape(-1, 64, 6491).cuda()
                # solution, recall, room_wise_acc, _ = run_ga(train_x_perturbed, model, fold)
                # print(solution)
                # train_x_perturbed = train_x_perturbed.reshape(ajr+1, m, 2*config.k_coefficient, -1)
                # embedding_1 = model1(train_x_perturbed[:,0,:,:])
                # embedding_2 = model2(train_x_perturbed[:,1,:,:])
                # embedding_3 = model3(train_x_perturbed[:,2,:,:])
                # embedding_4 = model4(train_x_perturbed[:,3,:,:])
                # embedding = torch.stack([embedding_1, embedding_2, embedding_3, embedding_4], dim=1).reshape(m*(ajr+1),-1)
                
                embedding = model(train_x_perturbed)
                # mining_func.mine(embedding, torch.LongTensor(train_y), embedding, torch.LongTensor(train_y)) # the smaller, the less triplets selected, the harder the problem is
                # embedding = model(torch.Tensor(train_x_reserved).cuda())
                
                # indices_tuple = mining_func(embedding, torch.Tensor(train_y))
                # print(embedding.shape)
                # print(torch.LongTensor(train_y))
                # print(torch.cat([torch.LongTensor(train_y), torch.full([40], 100)]).shape)
                # loss = criterion(embedding, torch.cat([torch.LongTensor(train_y), torch.full([40], 100)]))
                # loss = criterion(embedding, torch.cat([torch.Tensor(train_y), torch.Tensor(test_y)]))
                # loss = criterion(embedding, torch.Tensor(y))
                # print(embedding.shape)
                # print(type(batch_y))
                # print("batch_y:")
                # print(batch_y)
                loss = criterion(embedding, batch_y.reshape(-1, ).cuda())
                
                # print(loss) 
                train_losses.append(loss.detach().cpu().tolist())
                optimizer.zero_grad()           
                loss.backward()               
                optimizer.step()

                model.eval()
                test_embedding = model(torch.Tensor(test_x_reserved).cuda())
                test_loss = criterion(test_embedding, torch.LongTensor(test_y))
                test_losses.append(test_loss.detach().cpu().tolist())
                model.train()
                del loss, embedding, train_x_perturbed
                del test_embedding, test_loss
                torch.cuda.empty_cache()

            if epoch == config.epoch-1:
            # if epoch % 1000 == 0 or epoch == config.epoch-1:
                recall_list = []
                room_acc_list = []

                solution, recall, room_wise_acc, trip_acc = test_colocation(test_x_reserved, test_y, model, fold, args.split)
                recall_list.append(recall)
                room_acc_list.append(room_wise_acc)
                model.eval()
                cut_x = model(torch.tensor(test_x_reserved).cuda())
                # index_from_solution = torch.zeros(10,10)
                # for room in range(len(solution)):
                    
                # train_g = G.generate_graph(x=cut_x, index=torch.LongTensor(solution.reshape(-1,1)).cuda(), batch_size=1, cuda_flag=cuda_flag)
                # print(train_g)
                for _ in range(9):
                    solution, recall, room_wise_acc = run_ga(test_x_reserved, model, fold)
                    recall_list.append(recall)
                    room_acc_list.append(room_wise_acc)
                    print(solution)
                    # print(test_x_reserved.shape)
                    # train_g = G.generate_graph(x=cut_x, index=torch.LongTensor(solution.reshape(-1,1)).cuda(), batch_size=1, cuda_flag=cuda_flag)
                    # print(train_g)
                max_i = np.argmax(recall_list)
                mean_acc = np.mean(room_acc_list)
                max_acc = room_acc_list[max_i]
                triplet_accuracy[fold].append(trip_acc[0])
                room_accuracy[fold].append(max_acc)
                mean_room_accuracy[fold].append(mean_acc)
                # solution, recall, room_wise_acc = test_colocation(train_x_reserved[-40:], train_y[-40:], model, fold, args.split)

    print(triplet_accuracy)
    print(room_accuracy)
    print(mean_room_accuracy)

    with open("./ga.txt", "a+") as f:
        f.write("Triplet Accuracy: \n")
        for i in range(len(triplet_accuracy[0])):
            f.write("Epoch "+ str(10*i) + (": %s, " % np.asarray(triplet_accuracy)[:,i]))
            f.write("Mean: %s\n" % np.mean(np.asarray(triplet_accuracy)[:, i]))
        f.write("Room Accuracy: \n")
        for i in range(len(room_accuracy[0])):
            f.write("Epoch " + str(10*i) + (": %s, " % np.asarray(room_accuracy)[:, i]))
            f.write("Mean: %s\n" % np.mean(np.asarray(room_accuracy)[:, i]))
        f.write("Mean Room Accuracy: \n")
        for i in range(len(mean_room_accuracy[0])):
            f.write("Epoch " + str(10*i) + (": %s, " % np.asarray(mean_room_accuracy)[:, i]))
            f.write("Mean: %s\n" % np.mean(np.asarray(mean_room_accuracy)[:, i]))


if __name__ == '__main__':
    main()
    