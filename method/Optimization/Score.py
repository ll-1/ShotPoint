import numpy as np
from sklearn.cluster import MeanShift, DBSCAN
import copy
import matplotlib.pyplot as plt
from pyclustering.cluster.clique import clique
import time


def Offset_Distance(shot, receive):
    x1 = np.where(shot != 0)[0]
    y1 = np.where(shot != 0)[1]
    x2 = np.where(receive != 0)[0]
    y2 = np.where(receive != 0)[1]
    cmp_array = np.zeros_like(shot)
    for i in range(x1.shape[0]):
        for j in range(int(max(x1[i] - length_line, 0)), int(min(x1[i] + length_line, shot.shape[0]))):
            for k in range(int(max(y1[i] - dis_receive_line / dis_shot_point * height_line, 0)),
                           int(min(y1[i] + dis_receive_line / dis_shot_point * height_line, shot.shape[1]))):
                if receive[j, k] == 1:
                    ind_x = int((x1[i] + j) / 2)
                    ind_y = int((y1[i] + k) / 2)
                    cmp_array[ind_x, ind_y] += 1
    return cmp_array


def L1_residual(heatmap):
    [n, p] = heatmap.shape
    heatmap_sum = np.sum(heatmap)
    L1_score = np.sum(np.abs(heatmap / heatmap_sum - 1 / (n * p)))
    return L1_score


def monotonous(heatmap):
    [n, p] = heatmap.shape
    heatmap = heatmap / np.sum(heatmap)
    monotonous_score = 0
    for i in range(n):
        for j in range(p - 1):
            monotonous_score += abs(heatmap[i, j + 1] - heatmap[i, j])
    for j in range(p):
        for i in range(n - 1):
            monotonous_score += abs(heatmap[i + 1, j] - heatmap[i, j])
    return monotonous_score


def limitation(shot, boundary_matrix, offset_list, savepath):
    # center_list=copy.deepcopy(offset_list)
    # y_pred1=MeanShift(bandwidth=1).fit_predict(offset_list)
    # y_pred2 = DBSCAN(eps=0.1).fit_predict(offset_list)
    start = time.time()
    intervals = 100
    threshold = 1
    clique_instance = clique(offset_list, intervals, threshold)
    clique_instance.process()
    clique_cluster = clique_instance.get_clusters()  # allocated clusters

    end = time.time()
    # print(end - start)
    y_pred3 = np.zeros(len(offset_list))
    for i in range(y_pred3.shape[0]):
        for l in range(len(clique_cluster)):
            if i in clique_cluster[l]:
                y_pred3[i] = l
                break

    x2 = []
    y2 = []
    for i in offset_list:
        x2.append(i[0])
        y2.append(i[1])

    x1 = np.where(shot == 1)[0]
    y1 = np.where(shot == 1)[1]
    plt.figure(dpi=600)
    plt.scatter(x1, y1, s=3, color='red')
    plt.scatter(boundary_matrix[:, 0], boundary_matrix[:, 1], color='blue', s=3)

    plt.scatter(x2, y2, s=3, c=y_pred3)
    plt.savefig(savepath + 'cluster.png')
    plt.close()

    density_score = np.max(y_pred3) / len(offset_list)
    return density_score


def total_score(heatmap):
    return L1_residual(heatmap) + monotonous(heatmap)


def total_score_limitation(heatmap, shot, boundary_matrix, offset_list, savepath):
    a = L1_residual(heatmap)
    b = monotonous(heatmap)
    c = limitation(shot, boundary_matrix, offset_list, savepath)
    return 1/4*L1_residual(heatmap) + 1/4*monotonous(heatmap) + 2/4 * limitation(shot, boundary_matrix, offset_list, savepath)


def cmp_offset_distance(cmp):

    return


def cmp_direction_distribution(cmp):

    return
