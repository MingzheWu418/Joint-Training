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
from matplotlib import pyplot as plt
from sklearn.manifold import MDS, TSNE
import graph_handler
from ddqn import DQN
from graph_handler import GraphGenerator
from visualization import save_fig, save_fig_with_result, joint_visualization, dml_visualization
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

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-config', default = 'stn', type =str)
    parser.add_argument('-model', default='stn', type=str,
                        choices=['stn'])
    parser.add_argument('-pipeline', default='joint', type=str,
                        choices=['joint', 'two_step'])
    parser.add_argument('-seed', default=None, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='stn', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default=10606, type=int,
                        help="Log directory")
    parser.add_argument('-split',default='room', type=str,
                        help="split 1/5 sensors or rooms for test",
                        choices = ['room', 'sensor'])
    parser.add_argument('-synthetic', default=False, type=bool)
    parser.add_argument('--save_folder', default='/test')
    parser.add_argument('--train_distr', default='cluster', help="")
    parser.add_argument('--test_distr', default='cluster', help="")
    parser.add_argument('--target_mode', default=False)
    parser.add_argument('--num_rooms', default=50)
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
    parser.add_argument('--gpu', default='0', help="")
    parser.add_argument('--resume', default=False)
    parser.add_argument('--problem_mode', default='complete', help="")
    parser.add_argument('--readout', default='mlp', help="")
    parser.add_argument('--edge_info', default='adj_dist')
    parser.add_argument('--clip_target', default=0)
    parser.add_argument('--explore_method', default='epsilon_greedy') # epsilon_greedy
    parser.add_argument('--priority_sampling', default=0)
    parser.add_argument('--gamma', type=float, default=0.9, help="")
    parser.add_argument('--eps0', type=float, default=0.5, help="")
    parser.add_argument('--eps', type=float, default=0.1, help="")
    parser.add_argument('--explore_end_at', type=float, default=0.3, help="")
    parser.add_argument('--anneal_frac', type=float, default=0.7, help="")
    # 0.5 -> 0.1 -> 0.0
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--action_dropout', type=float, default=1.0)
    parser.add_argument('--n_epoch', default=20001)
    parser.add_argument('--save_ckpt_step', default=20000)
    parser.add_argument('--target_update_step', default=5)
    parser.add_argument('--replay_buffer_size', default=5000, help="") 
    parser.add_argument('--sample_batch_episode', type=int, default=0, help='')
    parser.add_argument('--ddqn', default=False)
    parser.add_argument('--visualization', default=True)
    parser.add_argument('--visualization_step', default=100)
    parser.add_argument('--save_log', default=True)

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
visualization = args.visualization
visualization_step = int(args.visualization_step)
save_log = args.save_log
synthetic = args.synthetic
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

def ri_to_label(k, m, room, coordinates):
    result = np.zeros(shape=(k*m, k))
    for i in range(len(coordinates)):
        for j in range(len(room)):
            if i in room[j]:
                result[i][j] = 1
    return result

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
        test_triplet = gen_colocation_triplet(test_x, test_y, prevent_same_type = True)
        test_loader = torch.utils.data.DataLoader(test_triplet, batch_size = 1, shuffle = False)
        cnt = 0
        test_anchor_list = []
        for step, batch_x in enumerate(test_loader):
            if args.model == 'stn':
                anchor = batch_x[0].cuda()
                pos = batch_x[1].cuda()
                neg = batch_x[2].cuda()
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
    
    test_out = out.cpu()
    test_corr = np.corrcoef(np.array(test_out))
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
    test_indexes = cross_validation_sample(args.num_rooms, k)
    '''read data & STFT'''
    logging(str(time.asctime( time.localtime(time.time()) )))
    if save_log:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './log/' + current_time + '/'

    if config.building in ["Soda", "SDH"]:
        print(config.building)
        x, y, true_pos = read_colocation_data(config.building, m, config)
    else:
        x, y, true_pos = read_colocation(config)
        if synthetic:
            ''' Generate synthetic data '''
            Fs_array = np.random.uniform(low=5000.0, high=20000.0, size=50)
            sample = 130000
            x_sin = np.arange(sample)
            y_sin = np.stack([200*np.sin(2 * np.pi * x_sin / Fs) for Fs in Fs_array])
            y_sin = np.repeat(y_sin, 4, axis=0)
            empty = torch.zeros(130000)
            for loop in range(200):
                if loop % 4 == 3:
                    y_sin[loop] = empty
            x = y_sin + x

    x = STFT(x, config)
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

        logging("Now training fold: %d" %(fold))
        # split training & testing
        print("Test indexes: ", test_index)
    
        train_x, train_y, test_x, test_y = split_colocation_train(x, y, test_index, args.split)
        train_x_reserved = np.stack(train_x, axis=0)
        test_x_reserved = np.stack(test_x, axis=0)

        # train_x = gen_colocation_triplet(train_x, train_y)
        total_triplets = len(train_x)
        logging("Total training triplets: %d\n" % (total_triplets))

        mean_q_err = []

        # We restart five times to find the best one during exploration
        # and then for the sixth trial we load the best model
        restart_array = [True] * 5
        restart_array.append(False)
        best_trial = -1
        
        for restart in range(len(restart_array)):
            # if torch.cuda.device_count() > 1:
            #     model = torch.nn.DataParallel(model)
            train_x = torch.Tensor(np.array(train_x)).reshape(-1, m, 64, 6491)
            train_y = torch.IntTensor(np.array(train_y)).reshape(-1, m)

            train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, drop_last=False)

            ''' Deep Metric Learning'''

            # load trained model
            # try:
            #     model.load_state_dict(torch.load("./output/DML_model_" + str(fold) + "_" + str(args.seed))) # DML
            #     for param in model.parameters():
            #         param.requires_grad = False
            # except FileNotFoundError:
            #     prints("No DML model found at the directory")

            if restart_array[restart]:
                # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # distance = LpDistance(power=2)
                train_losses = []
                test_losses = []
                
                if args.model == 'stn':
                    model = STN(config.dropout, 2 * config.k_coefficient).cuda()
                if config.optim == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, momentum = 0.9, weight_decay = config.weight_decay)
                elif config.optim == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

                ang_loss = AngularLoss()
                trip_loss = TripletMarginLoss(margin=1.0, distance=SNRDistance(), triplets_per_anchor=10) # 480 per anchor
                mining_func = MyMiner() # customized for 
                criterion = MultipleLosses([ang_loss, trip_loss], miners=[mining_func, mining_func])

                if config.grad_norm > 0:
                    nn.utils.clip_grad_value_(model.parameters(), config.grad_norm)
                    for p in model.parameters():
                        p.register_hook(lambda grad: torch.clamp(grad, -config.grad_norm, config.grad_norm))

                print("Model : ", model)
                print("Criterion : ", criterion)

                for epoch in range(config.epoch):
                    total_triplet_correct = 0

                    for step, (batch_x, batch_y) in enumerate(train_loader):
                        
                        x_in = batch_x.reshape(-1, 64, 6491).cuda()
                        embedding = model(x_in)
                        loss = criterion(embedding, batch_y.reshape(-1, ).cuda())
                        
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

                    if epoch == config.epoch-1:
                        solution, recall, room_wise_acc, _, _, _ = test_colocation(test_x_reserved, test_y, model, fold, args.split)
                        # solution, recall, room_wise_acc = test_colocation(train_x_reserved[-40:], train_y[-40:], model, fold, args.split)
                            
                    if epoch % visualization_step == 0:
                        
                        logging("Metric learning training epoch %d ......\n" % (epoch + 1))
                        if visualization:
                            # For every few steps, we visualize the learned embedding at the time,
                            # this is to make sure the learned embedding is reasonable
                            model.eval()
                            dml_visualization(model, train_x[:2*k*m], test_x_reserved, k, m, epoch, fold)
                            model.train()
                                
                iter_axis = np.arange(len(train_losses))
                plt.plot(iter_axis, train_losses, label='train')
                plt.plot(iter_axis, test_losses, label='test')
                plt.ylim(0,10)
                plt.savefig("./result/test_v_train" + str(int(time.time())) + ".png") 
                plt.clf()
                plt.close()
                for param in model.parameters():
                    param.requires_grad = True
            else:
                model = pickle.load(open(model_folder + "stn_"+ str(int(n_epoch * explore_end_at)) + "_restart_time_" + str(best_trial), 'rb'))
                model = model.cuda()
                for param in model.parameters():
                    param.requires_grad = True
                    
            # An embedding is learned, DQN
            RI_train_in = torch.Tensor(train_x_reserved).cuda()
            RI_test_in = torch.Tensor(test_x_reserved).cuda()

            if restart_array[restart]:
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
                                )
            else:
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
                                , model_file=model_folder + "/dqn_"+ str(int(n_epoch * explore_end_at)) + "_restart_time_" + str(best_trial)
                                )

            model_version = fold
            # global perturbation
            
            
            if restart_array[restart]:
                q_err_list = []
                total_epochs = int(n_epoch * explore_end_at)
            else:
                total_epochs = n_epoch-int(n_epoch * explore_end_at)

            for n_iter in tqdm(range(total_epochs)):

                model.train()
                if restart_array[restart]:
                    alg.eps = eps[n_iter]
                else:
                    alg.eps = eps[n_iter+int(n_epoch * explore_end_at)]
                
                RI_train_out = model(RI_train_in) # input -> DML -> output1
                RI_train_out = RI_train_out.cpu()

                '''Reinforcement Learning'''

                if n_iter % visualization_step == 0 and visualization:
                    # For every several epochs, we visualize the embedding to check the effect of joint-training
                    model.eval()
                    joint_visualization(model, RI_test_in, RI_train_in, k, m, n_iter, fold)
                    model.train()
                # print(id(model))
                # print(id(alg.DML_model))

                weights = torch.ones(args.num_rooms-10) 
                # number of training rooms, minus 10 because for our datasets we always save 10 rooms as the test rooms
                index = torch.cat([torch.multinomial(weights, k, replacement=False) for _ in range(n_episode)], 0)
                train_g = G.generate_graph(x=RI_train_out.detach(), index=index.reshape(-1, k), batch_size=n_episode, cuda_flag=cuda_flag)
                del RI_train_out
                torch.cuda.empty_cache()

                fine_tune = False
                if args.pipeline == 'joint' and n_iter % 10 == 0:
                    fine_tune = True
                
                # taking steps
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
                        , raw_x=RI_train_in
                        , fine_tune=fine_tune)
                if restart_array[restart]:
                    q_err_list.append(np.round(log.get_current('Q_error'), 3))

                if n_iter % target_update_step == target_update_step - 1:
                    alg.update_target_net()

                if not restart_array[restart]:
                    # once we have passed the exploration phase and selected the best one, 
                    # we can store trained model for future use
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

                # if save_log:
                #     train_summary_writer.add_scalar('Q_error', log.get_current('Q_error'), n_iter)
                #     train_summary_writer.add_scalar('S', log.get_current('tot_return'), n_iter)
                #     train_summary_writer.add_scalar('Trip_error', log.get_current('Triplet_error'), n_iter)

                if validation_step and (n_iter % validation_step == 0 or n_iter == total_epochs-1):
                    # validating the model

                    # We call the Genetic Algorithm method as a baseline
                    solution, recall, room_wise_acc, trip_acc, gt_fitness, ga_fitness = test_colocation(test_x, test_y, model, fold, args.split)

                    model.eval()
                    RI_test_out = torch.reshape(model(RI_test_in), (-1, 86))

                    test_corr = 1-np.corrcoef(np.array(RI_test_out.detach().cpu()))
                    X_transform = MDS(dissimilarity='precomputed').fit_transform(test_corr)
                    save_fig_with_result(k, m, 1, X_transform, solution, str(n_iter) + "GA_output")

                    avr_accuracy = []
                    avr_initial_value = []
                    final_value = []
                    gt_cluster_assignment = torch.nn.functional.one_hot(torch.arange(k).repeat_interleave(m).reshape(-1,1)).squeeze(1)
                    gt_test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag, label_input=gt_cluster_assignment)
                    print("Cut of groundTruth assignment: " + str(gt_test_g.kcut_value))
                    GA_cluster_assignment = torch.nn.functional.one_hot(torch.div(torch.Tensor(solution), m, rounding_mode="floor").reshape(-1,1).to(torch.int64))
                    GA_test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag, label_input=GA_cluster_assignment)
                    print("Cut of GA assignment: " + str(GA_test_g.kcut_value))

                    for _ in range(5):
                        # run five times and select the best

                        test_g = G.generate_graph(x=RI_test_out.cpu(), batch_size=test_episode, cuda_flag=cuda_flag)
                        # print(test_g)
                        
                        test = test_summary(alg=alg, graph_generator=G, action_type=action_type, q_net=readout, forbid_revisit=0)
                        test.run_test(problem=test_g, trial_num=1, batch_size=test_episode, action_dropout=action_dropout, gnn_step=gnn_step,
                                    episode_len=episode_len, explore_prob=0.0, Temperature=1e-8, cuda_flag=cuda_flag, save_log=True, k=k, m=m)
                        test.show_result(k, m)

                        # saving the result as figure
                        
                        # saving figures with scatter points and connected edges
                        # pca = PCA(n_components=2)
                        # X = PCA().fit_transform([RI_test_out].cpu().detach())
                        # save_fig(X, "test_graph_visualization" + str(n_iter))

                        RL_solution = m*torch.argmax(test.bg.ndata['label'], dim=1).reshape(1, k, m) + torch.arange(m).repeat(k).reshape(1,k,m).cuda()
                        RL_solution = RL_solution.detach().cpu().numpy()

                        corr_func = corr_score.compile_solution_func(1-test_corr, m)
                        fitnesses: np.ndarray = ga.fitness(RL_solution, corr_func)
                        save_fig_with_result(k, m, 1, X_transform, test.bg.ndata['label'].tolist()[:k*m], str(fold) + str(n_iter) + "RL_output" + str(_))

                        # calculate the accuracy
                        avr_accuracy.append(test.acc)
                        avr_initial_value.append(torch.mean(test.S).item())
                        final_value.append(np.float64(np.mean(test.bg.kcut_value.numpy())))

                    best_index = final_value.index(min(final_value))

                    # writing to a file and the tensorboard
                    if save_log:
                        with open(accuracy_file, 'a') as the_file:
                            the_file.write('Epoch: {}. Q error: {}. Trip Accuracy: {}. Best Accuracy: {} %. Best Final Value: {}. Mean Accuracy: {}. Std of Accuracy: {}. Mean Final Value: {}. Std of Final Value: {}. GA Accuracy: {}. GA Fitness: {}. GroundTruth Fitness: {}.'
                        .format(n_iter
                        , np.round(log.get_current('Q_error'), 3) # q_error
                        , np.round(trip_acc[-1], 3) # triplet acc
                        , np.round(avr_accuracy[best_index], 2) # best accuracy
                        , np.round(final_value[best_index], 2) # best final value
                        # , np.round(avr_initial_value[best_index], 2) # initial value corresponding to the best trial
                        , np.round(statistics.mean(avr_accuracy), 2) # mean accuracy
                        # , np.round(statistics.stdev(avr_accuracy), 2) # std of accuracy
                        , 0 # std of accuracy
                        , np.round(statistics.mean(final_value), 2) # mean final value
                        , 0 # std of final value
                        # , np.round(statistics.stdev(final_value), 2) # std of final value
                        , np.round(room_wise_acc, 2) # std of final value
                        , np.round(ga_fitness, 2)
                        , np.round(gt_fitness, 2)
                        ) + "\n")
                    
                    else:
                        print(avr_accuracy)

                    model.train()

                    RI_test_out = RI_test_out.cpu().detach()
                    del RI_test_out
                    del test_g
                    del test

                for node_attr in train_g.ndata.keys():
                    if node_attr in ('adj', 'label'):
                        train_g.ndata[node_attr] = train_g.ndata[node_attr].detach().cpu()
                del train_g
                del log

                torch.cuda.empty_cache()

            if restart_array[restart]:
                mean_q_err.append(np.mean(q_err_list))

                model_folder = './output/joint_training/' + str(current_time) + '/'
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                with open(model_folder + 'dqn_'+ str(n_iter + 1) + '_restart_time_' + str(restart), 'wb') as model_file:
                    pickle.dump(alg.model, model_file)
                with open(model_folder + 'stn_'+ str(n_iter + 1) + '_restart_time_' + str(restart), 'wb') as model_file:
                    pickle.dump(model, model_file)

                # The following part is used for model selection at the early phase,
                # in order to select the model that best explores the environment,
                best_trial = torch.multinomial(torch.nn.functional.softmax(torch.Tensor(mean_q_err)), 1).item()
                
            torch.cuda.empty_cache()

        RI_train_in = RI_train_in.cpu().detach()
        RI_test_in = RI_test_in.cpu().detach()
        model = model.cpu()
        criterion = criterion.cpu()
        del model, criterion
        del RI_train_in, RI_test_in
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    