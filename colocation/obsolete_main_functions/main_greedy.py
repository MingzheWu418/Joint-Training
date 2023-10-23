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
from envs import greedy_solver

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
# from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument('-seed', default=None, type=int,
                        help="Random seed")
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
    parser.add_argument('--gnn_step', default=3, help='')
    parser.add_argument('--test_batch_size', default=1, help='')
    parser.add_argument('--validation_step', default=200, help='')
    parser.add_argument('--gpu', default='1', help="")
    parser.add_argument('--resume', default=False)
    parser.add_argument('--problem_mode', default='complete', help="")
    parser.add_argument('--readout', default='mlp', help="")
    parser.add_argument('--edge_info', default='adj_dist')
    parser.add_argument('--clip_target', default=0)
    parser.add_argument('--explore_method', default='epsilon_greedy') # epsilon_greedy
    parser.add_argument('--priority_sampling', default=0)
    parser.add_argument('--gamma', type=float, default=0.9, help="")
    parser.add_argument('--eps0', type=float, default=0.1, help="") # 0.5
    parser.add_argument('--eps', type=float, default=0.1, help="") # 0.1
    parser.add_argument('--explore_end_at', type=float, default=0.4, help="") # 0.3
    parser.add_argument('--anneal_frac', type=float, default=0.6, help="") # 0.7
    # 0.5 -> 0.1 -> 0.0
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--action_dropout', type=float, default=1.0)
    parser.add_argument('--n_epoch', default=1001)
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
    out = model(x)
    test_out = out.detach().cpu()
    test_corr = np.corrcoef(np.array(test_out))

    scio.savemat('./output/corr_' + str(fold) + '.mat', {'corr':test_corr})
    acc_list = []
    best_solution, acc, ground_truth_fitness, best_fitness = run.ga(path_m = './output/corr_' + str(fold) + '.mat', path_c = '10_rooms.json')
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
    return best_solution, recall, room_wise_acc, trip_acc, ground_truth_fitness, best_fitness


class MyMiner(BaseTupleMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)
        type_constraint = torch.logical_and(n%m != a%m, n%m != p%m)
        triplet_mask = torch.logical_and(type_constraint, a != p)
        # random_mask = torch.randint(0,5, (triplet_mask.shape)).cuda()
        # triplet_mask = torch.logical_and(triplet_mask, random_mask > 3)
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]


def main():
    G = GraphGenerator(k=k, m=m, ajr=ajr, style=train_graph_style)
    # test_indexes = cross_validation_sample(50, k)
    test_indexes = cross_validation_sample(37, k)
    # test_indexes =  [[38, 10, 24, 35, 0], [45, 26, 9, 29, 16], [46, 36, 32, 44, 13], [25, 23, 19, 11, 4], [31, 21, 12, 3, 39], [43, 18, 33, 48, 41], [30, 28, 20, 22, 42], [49, 47, 2, 27, 37], [5, 34, 6, 8, 14], [15, 17, 1, 7, 40]]
    '''read data & STFT'''
    logging(str(time.asctime( time.localtime(time.time()) )))
    if save_log:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './log/' + current_time + '/'
    # print(config.building)
    if config.building in ["Soda", "SDH"]:
        print(config.building)
        x, y, true_pos = read_colocation_data(config.building, m, config)
    else:
        x, y, true_pos = read_colocation(config)
    # x, y, true_pos = x[:64], y[:64], true_pos[:64]
    # print("===== x shape =====")
    # print(np.asarray(x).shape)

    ''' Synthetic data '''
    # Fs_array = np.random.uniform(low=5000.0, high=20000.0, size=50)
    # sample = 130000
    # x_sin = np.arange(sample)
    # y_sin = np.stack([200*np.sin(2 * np.pi * x_sin / Fs) for Fs in Fs_array])
    # y_sin = np.repeat(y_sin, 4, axis=0)
    # empty = torch.zeros(130000)
    # for loop in range(200):
    #     if loop % 4 == 3:
    #         y_sin[loop] = empty
    # print(y_sin)
    # x = y_sin + x

    x = STFT(x, config)
    # print(np.asarray(x).shape)
    # print(y)
    logging("%d total sensors, %d frequency coefficients, %d windows\n" % (len(x), x[0].shape[0], x[0].shape[1]))

    print("test indexes:\n", test_indexes)

    fold_recall = []
    fold_room_acc = []

    triplet_accuracy = []
    room_accuracy = []

    """ 5-fold cross validation"""
    # 5 folds
    # 40 as training, 10 testing
    for fold, test_index in enumerate(test_indexes):
        if save_log:
            tensorboard_dir = './log/' + current_time + 'fold' + str(fold) + '/'
            # train_summary_writer = SummaryWriter(tensorboard_dir)
            accuracy_dir = './result/accuracy/RealTrainingRealTesting/' + train_log_dir + '/fold' + str(fold) + '_pipeline/'
            if not os.path.exists(accuracy_dir):
                os.makedirs(accuracy_dir)
            accuracy_file = accuracy_dir + 'accuracy.txt'
            with open(accuracy_file, 'w+') as the_file:
                the_file.write("Learning Rate: " + str(lr) + "\n")
                the_file.write("Action Dropout: " + str(action_dropout) + "\n")
                the_file.write("Replay Buffer Size: " + str(replay_buffer_size) + "\n")
                the_file.write("Number of Episodes: " + str(n_episode) + "\n")
                the_file.write("Exploration rate: " + str(explore_end_at) + "\n")
                the_file.write("Dropout rate: " + str(config.dropout) + "\n")
                the_file.write("---------- Now Training Fold:" + str(fold) + " ----------\n")
        
        triplet_accuracy.append([])
        room_accuracy.append([])
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

        mean_q_err = []

                
        if args.loss == 'triplet':
            criterion = tripletLoss(margin = 1).cuda()
        elif args.loss == 'comb':
            criterion = combLoss(margin = 1).cuda()

        if args.model == 'stn':
            model = STN(config.dropout, 2 * config.k_coefficient).cuda()
            # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(pytorch_total_params)
            
        if config.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, momentum = 0.9, weight_decay = config.weight_decay)
        elif config.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

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

        distance = LpDistance(power=2)
        # distance = CosineSimilarity()
        ang_loss = AngularLoss()
        trip_loss = TripletMarginLoss(margin=1.0, distance=SNRDistance(), triplets_per_anchor=100) # 480 per anchor
        mining_func = MyMiner()
        criterion = MultipleLosses([ang_loss, trip_loss], miners=[mining_func, mining_func])
        # mining_func = BatchHardMiner()
        # criterion = MultipleLosses([trip_loss], miners=[mining_func])
        # criterion = MultipleLosses([ang_loss, trip_loss])
        # criterion = trip_loss
        # test_x_vali = torch.Tensor(np.stack(x[160:], axis=0)).cuda()
        train_losses = []
        test_losses = []
    
        for epoch in range(config.epoch):

            if epoch % 100 == 0:
                logging("Now training %d epoch ......\n" % (epoch + 1))
            total_triplet_correct = 0

            for step, (batch_x, batch_y) in enumerate(train_loader):
                
                x_in = batch_x.reshape(-1, 64, 6491).cuda()
                embedding = model(x_in)
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
                del loss, embedding, x_in
                del test_embedding, test_loss
                torch.cuda.empty_cache()
        
        print(test_losses)

        RI_train_in = torch.Tensor(train_x_reserved).cuda()
        RI_test_in = torch.Tensor(test_x_reserved).cuda()

        solution, recall, room_wise_acc, trip_acc, _, _ = test_colocation(test_x_reserved, test_y, model, fold, args.split)
        
        RI_test_out = torch.reshape(model(RI_test_in), (-1, 86))
        test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag)
        print("===================================")
        # print(test_g)
        initial_room_assignment = test_g.ndata['label'].detach().cpu()
        # print(initial_room_assignment)
        print((initial_room_assignment == 1).nonzero()[:, 1].reshape(k, m))
        final_g, _, rewards = greedy_solver(test_g, step=100)
        final_room_assignment = final_g.ndata['label'].detach().cpu()
        # print(final_room_assignment)
        print((final_room_assignment == 1).nonzero()[:, 1].reshape(k, m))
        print("===================================")
                        
        RI_train_in = RI_train_in.cpu().detach()
        RI_test_in = RI_test_in.cpu().detach()
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
    
    # print(triplet_accuracy)
    # print(room_accuracy)

    # with open("./ga_correct_ml_all1000epochs.txt", "w+") as f:
    #     f.write("Triplet Accuracy: \n")
    #     for item in triplet_accuracy:
    #         f.write("%s\n" % item)
    #     f.write("Room Accuracy: \n")
    #     for item in room_accuracy:
    #         f.write("%s\n" % item)


if __name__ == '__main__':
    main()
    