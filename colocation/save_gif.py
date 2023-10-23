
import imageio

for i in range(5):
    gif_fig_list = []
    gif_fig_list_train = []
    for n_iter in range(21):
        gif_fig_list.append("./result/" + str(22000+100*n_iter) + "_test_" + str(i)+ ".png")
        gif_fig_list_train.append("./result/" + str(22000+100*n_iter) + "_train_" + str(i)+ ".png")

    ims = [imageio.imread(f) for f in gif_fig_list]
    imageio.mimwrite("./result/test" + str(i) +".gif", ims, fps=5)

    ims_train = [imageio.imread(f) for f in gif_fig_list_train]
    imageio.mimwrite("./result/train" + str(i) +".gif", ims_train, fps=5)
