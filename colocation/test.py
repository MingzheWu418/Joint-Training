import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import utils.util as utils
from Data import *
import matplotlib.colors as colors
import scipy.fftpack


# Fixing random state for reproducibility
np.random.seed(19680801)

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
    parser.add_argument('--eps0', type=float, default=0.0, help="") # 0.5
    parser.add_argument('--eps', type=float, default=0.0, help="") # 0.1
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

x, y, true_pos = read_colocation(config)

# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = 10 + np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# print(y.shape)
# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# plt.subplot(2, 1, 1)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
# plt.subplot(2, 1, 2)
# plt.plot(xf[1:], 2.0/N * np.abs(yf[0:N/2])[1:])

# # Number of samplepoints
# N = 400
# # sample spacing
# T = 1.0 / 3000.0
# y = np.array(x)[:8]
# # Fs_array = np.random.uniform(low=50000.0, high=50000.0, size=1)
# # print(Fs_array)
# # x_sin = np.arange(N)
# # y_sin = np.stack([0*np.sin(2 * np.pi * x_sin / Fs) for Fs in Fs_array])
# # y = y_sin[0] + y
# print(np.array(y).shape)
# x = np.linspace(0.0, N)
# yf = scipy.fft.fft(y)
# print(x.shape)
# print(yf)
# print(yf.shape)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2-100)
# print(xf)
# print(xf.shape)
# fig, ax = plt.subplots()
# ax.plot(xf, np.abs(yf[:N//2]))
# plt.show()

Fs_array = np.random.uniform(low=5000.0, high=20000.0, size=50)
sample = 130000
x_sin = np.arange(sample)
y_sin = np.stack([200*np.sin(2 * np.pi * x_sin / Fs) for Fs in Fs_array])
y_sin = np.repeat(y_sin, 4, axis=0)
empty = torch.zeros(130000)
for loop in range(200):
    if loop % 4 == 3:
        y_sin[loop] = empty
print(y_sin)

# x = y_sin + x

fig, ax = plt.subplots()

# The ticks
plt.axis("off")
plt.figure(figsize=(30,6))
plt.xticks([])
plt.yticks([])
plt.plot(np.arange(130000), x[0], linewidth=3)
plt.xlim(0,130000)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
plt.savefig("./test.png")
# plt.clf()
# print(np.asarray(x).shape)
# # # x, y, true_pos = x[:64], y[:64], true_pos[:64]
# # # print("===== x shape =====")
# # # print(np.asarray(x).shape)
# stft = STFT(x, config)
# print(np.asarray(stft).shape)
# data = np.asarray(x)
# # data = np.random.normal(0, 1000, (1, 64, 12981))
# # print(data.shape)
# # Create new Figure with black background
# fig = plt.figure(figsize=(200, 300), facecolor='black')

# # Add a subplot with no frame
# ax = plt.subplot(111, frameon=False)

# # Generate random data

# y = np.linspace(1,64, 64)
# x = np.linspace(1,12981, 12981)
# z = data[0] - np.amin(data[0]) + 1
# # z = np.outer(y,x)
# print(x)
# print(y)
# print(z.shape)
# print(np.min(z), np.max(z))

# bounds = [np.amin(z), np.amax(z)]
# bounds = np.log10(bounds)
# bounds[0] = np.floor(bounds[0])
# bounds[1] = np.ceil(bounds[1])
# bounds = np.power(10, bounds)

# fig, ax = plt.subplots()
# CS = ax.pcolormesh(x, y, z, norm=colors.LogNorm(*bounds), shading="auto")
# cbar = plt.colorbar(CS, ax=ax)
# print("Boundary values")
# print(bounds)
# print("Tick values")
# print(cbar.get_ticks())

# for i in range(200):
#     plt.plot(np.arange(len(x[i])), x[i])
#     plt.show()
#     plt.savefig("./test" + str(i//4) + "_" + str(i%4) + ".png")
#     plt.clf()