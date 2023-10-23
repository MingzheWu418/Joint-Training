import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import torch
import numpy as np


def save_fig(dataset, filename, m):
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
            x_list[room_number+k*j].append(x[m*k*j+l][0])
            y_list[room_number+k*j].append(x[m*k*j+l][1])
    colors = cm.rainbow(np.linspace(0, 1, len(x_list)))

    # The scatter points represents which points are really in the same groups
    # the plot represents which points are predicted to be in the same groups
    for x, y, x_original, y_original, c in zip(x_list, y_list, print_x, print_y, colors):
        plt.scatter(x_original, y_original, color=c)
        plt.plot(x, y, color=c)

    plt.savefig("./result/RI/sim_output" + str(n_iter) + ".png")
    plt.clf()
    plt.close()

def joint_visualization(model, test_embedding, train_embedding, k, m, n_iter, fold):
    RI_test_out_visualization = torch.reshape(model(test_embedding), (-1, 86))
    X_transform = PCA().fit_transform(np.array(RI_test_out_visualization.detach().cpu()))
    save_fig(X_transform, str(n_iter+20000)+"_test_" + str(fold), m)
    RI_test_out_visualization = RI_test_out_visualization.detach().cpu()
    del RI_test_out_visualization
    torch.cuda.empty_cache()

    RI_train_out_visualization = model(train_embedding)
    X_transform = PCA().fit_transform(np.array(RI_train_out_visualization.detach().cpu()[k*m:2*k*m]))
    save_fig(X_transform, str(n_iter+20000)+"_train_" + str(fold), m)
    del RI_train_out_visualization
    torch.cuda.empty_cache()

def dml_visualization(model, train_x, test_x, k, m, epoch, fold):
    test_out = model(torch.Tensor(test_x).cuda()).cpu()
    train_out = model(train_x.reshape(-1, 64, 6491).cuda()).cpu()

    X_test_transform = PCA(random_state=42).fit_transform(test_out.detach().cpu())
    save_fig(X_test_transform, str(fold+1) + "_" + str(epoch+1) + "test_pretrain", m)

    X_transform = PCA(random_state=42).fit_transform(torch.cat([train_out.detach(), test_out.detach()]))
    del test_out
    torch.cuda.empty_cache()
    
    X_train_transform = X_transform[:-k*m]
    X_test_transform = X_transform[-k*m:]
    '''Saving a figure with scattered points'''
    print_x = []
    print_y = []

    for i in range(len(X_train_transform)//m):
        print_x.append([])
        print_y.append([])

    for index, item in enumerate(X_train_transform):
        # print(index)
        print_x[index//m].append(float(item[0]))
        print_y[index//m].append(float(item[1]))

    colors = cm.rainbow(np.linspace(0, 1, len(print_y)))
    for a, b, c in zip(print_x, print_y, colors):
        plt.scatter(a, b, color=c, marker="o")
        # plt.plot(x, y, color=c)

    print_x = []
    print_y = []

    for i in range(len(X_test_transform)//m):
        print_x.append([])
        print_y.append([])

    for index, item in enumerate(X_test_transform):
        # print(index)
        print_x[index//m].append(float(item[0]))
        print_y[index//m].append(float(item[1]))

    colors = cm.rainbow(np.linspace(0, 1, len(print_y)))
    for a, b, c in zip(print_x, print_y, colors):
        plt.scatter(a, b, color=c, marker="D")
    
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig("./result/" + str(fold+1) + "_" + str(epoch+1) + "train_and_test" + ".png")
    plt.clf()
    plt.close()

    save_fig(X_train_transform, str(fold+1) + "_" + str(epoch+1) + "train_pretrain_10rooms", m)
    del train_out
