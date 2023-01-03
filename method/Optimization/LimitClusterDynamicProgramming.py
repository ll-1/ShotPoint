import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import seaborn as sns
import os
import heapq
import matplotlib
import matplotlib.pyplot as plt
from pyclustering.cluster.clique import clique, clique_visualizer
import sys
sys.path.append('./')
from method.Optimization.GraphPartition import cmp_array
from utils.CMP import adapative_search_nearest_neighbor, limit_adapative_search_nearest_neighbor

matplotlib.use('Agg')


def expand_obstacle(obstacle, x, y, safe_radius, shotbinsize, receivebinsize):
    for i in range(int(safe_radius / receivebinsize) + 1):
        for j in range(int(safe_radius / shotbinsize) + 1):
            if (i * receivebinsize) ** 2 + (j * shotbinsize) ** 2 <= safe_radius ** 2:
                obstacle[x + i, y - j] = 1
                obstacle[x - i, y + j] = 1
                obstacle[x + i, y + j] = 1
                obstacle[x - i, y - j] = 1
    return obstacle


def BackwardOptimization(shot, shotframe, receive, pre_obstacle, offset_obstacle, patchdict,
                         start_receive_line_position,
                         start_receive_point_position,
                         dis_shot_line, dis_shot_point, dis_receive_line, dis_receive_point, min_dis_shot, shotbinsize,
                         receivebinsize, para_list, punish_lambda,
                         max_radius, safe_radius,
                         path, Patchpath,
                         savepath, distance,
                         congruence=True, set_region=True, k=10, min_goal='var', line_punishment=False,
                         limit_dis_shot_point=False, use_prior=False, expand_boundary=10, stop=False):
    predframe = copy.deepcopy(shotframe)
    total_shot_number = shot.shape[0] * shot.shape[1]
    if use_prior:
        for ind in range(shotframe.shape[0]):
            if shotframe.loc[ind, 'can_not_move'] == 1:
                pre_obstacle = expand_obstacle(pre_obstacle, int(shotframe.loc[ind, 'shot_line_position']),
                                               int(shotframe.loc[ind, 'shot_point_position']), 1.5 * shotbinsize,
                                               shotbinsize, receivebinsize)
    in_shot = shot * pre_obstacle
    print(np.sum(in_shot))
    if use_prior:
        for ind in range(shotframe.shape[0]):
            if shotframe.loc[ind, 'can_not_move'] == 1:
                in_shot[
                    int(shotframe.loc[ind, 'shot_line_position']), int(shotframe.loc[ind, 'shot_point_position'])] = 0
    print(np.sum(in_shot))
    out_shot = shot - in_shot
    print(np.sum(out_shot))

    keep_same_ind = []
    aaa = np.where(out_shot != 0)[0]
    bbb = np.where(out_shot != 0)[1]
    for ind in range(len(aaa)):
        xxx = aaa[ind]
        yyy = bbb[ind]
        lineind = shotframe[(shotframe['shot_line_position'] == xxx) & (
                shotframe['shot_point_position'] == yyy)].index.tolist()
        keep_same_ind.append(lineind[0])

    n, p = shot.shape
    min_i = 0
    min_j = 0
    max_i = n
    max_j = p
    if set_region:
        min_i = max(sorted(set(np.where(shot != 0)[0]))[0] - expand_boundary, 0)
        min_j = max(sorted(set(np.where(shot != 0)[1]))[0] - expand_boundary, 0)
        max_i = min(sorted(set(np.where(shot != 0)[0]))[-1] + 1 + expand_boundary, n)
        max_j = min(sorted(set(np.where(shot != 0)[1]))[-1] + 1 + expand_boundary, p)

    score_list = []

    if not os.path.exists(path + str(safe_radius) + '_' + str(use_prior) + '_' + 'init_cmp.npy'):
        out_cmp = cmp_array(out_shot, receive, patchdict, path, path, dis_receive_line, dis_receive_point)
        np.save(path + str(safe_radius) + '_' + str(use_prior) + '_' + 'init_cmp.npy', out_cmp)
        np.save(savepath + 'init_cmp.npy', out_cmp)
    else:
        out_cmp = np.load(path + str(safe_radius) + '_' + str(use_prior) + '_' + 'init_cmp.npy', allow_pickle=True)
        np.save(savepath + 'init_cmp.npy', out_cmp)
    init_cmp = copy.deepcopy(out_cmp)
    vmax = np.max(out_cmp)
    cmap = 'rainbow'
    sns.heatmap(out_cmp, annot=False, vmax=vmax, vmin=0, cmap=cmap)
    plt.savefig(savepath + 'init_cmp.png')
    # plt.show()
    plt.close()

    out_mu = np.mean(out_cmp)
    out_Var = np.var(out_cmp)
    score_list.append(out_Var)

    # offset the clustered shot points
    f1 = np.where(in_shot != 0)[0].reshape(-1, 1)
    f2 = np.where(in_shot != 0)[1].reshape(-1, 1)

    minx = np.min(f1)
    maxx = np.max(f1)
    miny = np.min(f2)
    maxy = np.max(f2)

    # Cluster the source points
    shotlist = (np.hstack((f1, f2)) - np.array([[minx, miny]])) / dis_shot_point
    clique_instance = clique(shotlist, amount_intervals=int((maxy - miny) / dis_shot_point),
                             density_threshold=0)
    clique_instance.process()
    clique_cluster = clique_instance.get_clusters()

    shotlist = np.array(shotlist * dis_shot_point + np.array([[minx, miny]]), dtype=int)

    not_move_ind = []
    with tqdm(total=len(clique_cluster)) as pbar:
        for i in range(len(clique_cluster)):
            clique_shot_ind = clique_cluster[i]
            clique_shot = []
            for ind in clique_shot_ind:
                in_point = shotlist[ind, :]
                clique_shot.append([in_point[0], in_point[1]])

            clique_shot = np.array(clique_shot)
            shotframeind = []
            shot_line_score = dict()

            candidate_out_shot = copy.deepcopy(out_shot)

            # Define the neighbor set for every source point
            clique_list = []

            for t in range(clique_shot.shape[0]):
                ind = t
                or_point = clique_shot[ind, :]

                lineind = shotframe[(shotframe['shot_line_position'] == or_point[0]) & (
                        shotframe['shot_point_position'] == or_point[1])].index.tolist()
                shotframeind.append(lineind[0])

                candidate_list = []
                cand_score_list = []

                if limit_dis_shot_point:
                    first_list = limit_adapative_search_nearest_neighbor(
                        (1 - candidate_out_shot) * (1 - offset_obstacle),
                        candidate_out_shot,
                        or_point[0],
                        or_point[1], k, max_radius, min_i, max_i, min_j,
                        max_j, min_dis_shot, shotbinsize, receivebinsize, distance, stop)
                else:
                    first_list = adapative_search_nearest_neighbor((1 - candidate_out_shot) * (1 - offset_obstacle),
                                                                   or_point[0],
                                                                   or_point[1], k, max_radius, min_i, max_i, min_j,
                                                                   max_j, shotbinsize, receivebinsize, distance)
                for point in first_list:
                    if min_i <= point[0] < max_i and min_j <= point[1] < max_j and out_shot[
                        point[0], point[1]] == 0:
                        candidate_list.append(point)

                if candidate_list == []:
                    not_move_ind.append(lineind)
                    candidate_list.append([or_point[0], or_point[1]])
                for point in candidate_list:
                    candidate_out_shot[point[0], point[1]] = 1

                clique_list.extend(candidate_list)

                for candidate_point in candidate_list:
                    x = candidate_point[0]
                    y = candidate_point[1]
                    if congruence:
                        if not os.path.exists(Patchpath + 'NotCoverage/' + str(x) + '_' + str(y) + '.npy'):
                            congruencex = int((x - start_receive_point_position) % dis_shot_line)
                            congruencey = int((y - start_receive_line_position) % dis_receive_line)
                            readpath = Patchpath + 'Coverage/' + str(congruencex) + '_' + str(congruencey) + '.npy'
                            direct_cmp = np.load(readpath, allow_pickle=True)
                            direct_cmp = direct_cmp.item()[(congruencex, congruencey)]
                        else:
                            direct_cmp = np.load(Patchpath + 'NotCoverage/' + str(x) + '_' + str(y) + '.npy',
                                                 allow_pickle=True)
                            direct_cmp = direct_cmp.item()[(x, y)]
                        start_x = x - direct_cmp[0]
                        start_y = y - direct_cmp[1]
                        xx = np.where(direct_cmp[4] != 0)[0]
                        yy = np.where(direct_cmp[4] != 0)[1]
                        try:
                            cand_score_list.append((2 * np.sum(out_cmp[start_x + xx, start_y + yy]) + len(xx) * (
                                    total_shot_number - len(xx)) / total_shot_number - 2 * out_mu * len(
                                xx)) / total_shot_number)
                        except:
                            print(x, y)
                    else:
                        x = candidate_point[0]
                        y = candidate_point[1]
                        direct_cmp = np.load(Patchpath + str(x) + '_' + str(y) + '.npy', allow_pickle=True)
                        direct_cmp = direct_cmp.item()[(x, y)]
                        start_x = direct_cmp[0]
                        start_y = direct_cmp[1]
                        xx = np.where(direct_cmp[4] != 0)[0]
                        yy = np.where(direct_cmp[4] != 0)[1]
                        try:
                            cand_score_list.append((2 * np.sum(out_cmp[start_x + xx, start_y + yy]) + len(xx) * (
                                    total_shot_number - len(xx)) / total_shot_number - 2 * out_mu * len(
                                xx)) / total_shot_number)
                        except:
                            print(x, y)

                shot_line_score[(or_point[0], or_point[1])] = []
                min_k_index = [i for i in range(min(k, len(cand_score_list)))]
                for ind in min_k_index:
                    off_pos = candidate_list[ind]
                    off_var = cand_score_list[ind]
                    shot_line_score[(or_point[0], or_point[1])].append([off_pos, off_var])


            shotframeind = list(reversed(shotframeind))

            backwardscorearray = np.zeros((clique_shot.shape[0], k))
            backwardindarray = np.zeros_like(backwardscorearray)
            backwardvararray = np.zeros_like(backwardscorearray)
            backwardlinedisarray = np.zeros_like(backwardscorearray)
            backwardlineazimutharray = np.zeros_like(backwardscorearray)

            # 使用后向优化确定最优点集
            min_parameter_score = 10000000
            for [a0, a1] in para_list:
                ind = 0
                cover_point_list = False
                for key, value in shot_line_score.items():
                    j = 0
                    if ind == 0:
                        last_pos = []
                        for point in value:
                            linedis = 0
                            lineazimuth = 0
                            if key[1] >= dis_shot_point:
                                linedis = math.fabs(point[0][1] - (key[1] - 4))
                                if point[0][1] == key[1]:
                                    lineazimuth = 10
                                else:
                                    lineazimuth = math.fabs(point[0][0] - (key[0] - 4)) / math.fabs(
                                        point[0][1] - key[1])
                            backwardscorearray[ind, j] = a0 * point[1] + a1 * linedis + (
                                    1 - a0 - a1) * lineazimuth
                            backwardvararray[ind, j] = point[1]
                            backwardlinedisarray[ind, j] = linedis
                            backwardlineazimutharray[ind, j] = lineazimuth
                            backwardindarray[ind, j] = j
                            last_pos.append(point[0])
                            j += 1
                            if j == len(value):
                                backwardscorearray[ind, j:] = 1000000
                                backwardindarray[ind, j:] = j
                                break
                    else:
                        copy_last_pos = copy.deepcopy(last_pos)
                        last_pos = []
                        for point in value:
                            pos_dis = []
                            for tt in range(len(copy_last_pos)):
                                linedis = math.fabs(copy_last_pos[tt][1] - point[0][1])
                                if point[0][1] == copy_last_pos[tt][1]:
                                    lineazimuth = 10
                                else:
                                    lineazimuth = math.fabs(copy_last_pos[tt][0] - point[0][0]) / math.fabs(
                                        copy_last_pos[tt][1] - point[0][1])
                                pos_dis.append(a0 * point[1] + a1 * linedis + (1 - a0 - a1) * lineazimuth)
                            bestind = np.argmin(pos_dis)
                            backwardscorearray[ind, j] = pos_dis[bestind]
                            backwardvararray[ind, j] = backwardvararray[ind - 1, bestind] + point[1]
                            backwardlinedisarray[ind, j] = backwardlinedisarray[ind - 1, bestind] + linedis
                            backwardlineazimutharray[ind, j] = backwardlinedisarray[
                                                                   ind - 1, bestind] + lineazimuth
                            backwardindarray[ind, j] = bestind
                            last_pos.append(point[0])
                            j += 1
                            if j == len(value):
                                backwardscorearray[ind, j:] = 1000000
                                backwardindarray[ind, j:] = j
                                break
                    ind += 1
                last_point = []
                for backward_ind in range(clique_shot.shape[0] - 1, -1, -1):
                    if backward_ind == clique_shot.shape[0] - 1:
                        best_backward_ind = np.argmin(backwardscorearray[backward_ind, :])
                        last_point.append(
                            list(shot_line_score.values())[backward_ind][best_backward_ind][0])
                        if min_goal == 'var':
                            if backwardvararray[backward_ind, best_backward_ind] < min_parameter_score:
                                min_parameter_score = backwardvararray[backward_ind, best_backward_ind]
                                cover_point_list = True
                        elif min_goal == 'line_dis':
                            if backwardlinedisarray[backward_ind, best_backward_ind] < min_parameter_score:
                                min_parameter_score = backwardlinedisarray[backward_ind, best_backward_ind]
                                cover_point_list = True
                        elif min_goal == 'offset_dis':
                            if backwardlineazimutharray[backward_ind, best_backward_ind] < min_parameter_score:
                                min_parameter_score = backwardlineazimutharray[backward_ind, best_backward_ind]
                                cover_point_list = True
                        else:
                            raise Exception('Not correct goal.')
                    else:
                        # print(backward_ind)
                        best_backward_ind = int(backwardindarray[backward_ind + 1, best_backward_ind])
                        last_point.append(
                            list(shot_line_score.values())[backward_ind][best_backward_ind][0])
                if cover_point_list:
                    last_offset_point = copy.deepcopy(last_point)

            for pointind in range(len(last_offset_point)):
                point = last_offset_point[pointind]
                x = point[0]
                y = point[1]
                if congruence:
                    if not os.path.exists(Patchpath + 'NotCoverage/' + str(x) + '_' + str(y) + '.npy'):
                        congruencex = int((x - start_receive_point_position) % dis_shot_line)
                        congruencey = int((y - start_receive_line_position) % dis_receive_line)
                        readpath = Patchpath + 'Coverage/' + str(congruencex) + '_' + str(congruencey) + '.npy'
                        direct_cmp = np.load(readpath, allow_pickle=True)
                        direct_cmp = direct_cmp.item()[(congruencex, congruencey)]
                    else:
                        direct_cmp = np.load(Patchpath + 'NotCoverage/' + str(x) + '_' + str(y) + '.npy',
                                             allow_pickle=True)
                        direct_cmp = direct_cmp.item()[(x, y)]
                    out_cmp[int(x - direct_cmp[0]):int(x + direct_cmp[2]),
                    int(y - direct_cmp[1]): int(y + direct_cmp[3])] += direct_cmp[4]
                    out_shot[x, y] += 1
                else:
                    direct_cmp = np.load(Patchpath + str(x) + '_' + str(y) + '.npy', allow_pickle=True)
                    direct_cmp = direct_cmp.item()[(x, y)]
                    out_cmp[int(direct_cmp[0]):int(direct_cmp[2]), int(direct_cmp[1]): int(direct_cmp[3])] += \
                        direct_cmp[4]
                    out_shot[x, y] += 1

                predframe.loc[shotframeind[pointind], 'shot_line_position'] = x
                predframe.loc[shotframeind[pointind], 'shot_point_position'] = y
                out_mu = np.mean(out_cmp)
                out_Var = np.var(out_cmp)
                score_list.append(out_Var)

                f = open(savepath + 'parameter.txt', 'a')
                f.writelines([str(x) + ' ', str(y) + ' ', str(score_list[-1]) + ' ', '\n'])
                f.close()

            plt.figure(dpi=100)
            plt.scatter(np.where(out_shot == 1)[0], np.where(out_shot == 1)[1], s=0.3, color='red')
            plt.savefig(savepath + 'dynamic_out_shot.png')
            plt.close()

            pbar.set_description('offset the shot line')
            pbar.update(1)
    predframe['not_move'] = '0'
    for ind in not_move_ind:
        predframe.loc[ind, 'not_move'] = 1
    predframe['in_obstacle'] = '0'
    for ind in range(predframe.shape[0]):
        if offset_obstacle[
            int(predframe.loc[ind, 'shot_line_position']), int(predframe.loc[ind, 'shot_point_position'])] == 1:
            predframe.loc[ind, 'in_obstacle'] = 1
        else:
            predframe.loc[ind, 'in_obstacle'] = 0

    predframe.to_csv(savepath + 'offshotframe.csv', index=False)
    print('The difference of CMP between offset：{}'.format(np.sum(np.abs(out_cmp - init_cmp))))
    return out_shot, score_list, out_cmp, keep_same_ind


def LimitClusterBackwardOptimizationmain(orpath, offpath, Patchpath, max_radius, safe_radius, congruence, set_region,
                                         min_goal,
                                         savedir, line_punishment, para_list, punish_lambda, limit_dis_shot_point,
                                         min_dis_shot,
                                         use_prior, distance, expand_boundary, stop):
    if use_prior:
        shot = np.load(orpath + 'regularpriorshotarray.npy', allow_pickle=True)
        shotframe = pd.read_csv(orpath + 'regularpriorshotframe.csv', index_col=False)
    else:
        shot = np.load(orpath + 'regularshotarray.npy', allow_pickle=True)
        shotframe = pd.read_csv(orpath + 'regularshotframe.csv', index_col=False)
    receive = np.load(orpath + 'regularreceivearray.npy', allow_pickle=True)
    pre_obstacle = np.load(orpath + 'pre_obstacle.npy', allow_pickle=True)
    offset_obstacle = np.load(orpath + 'offset_obstacle.npy', allow_pickle=True)
    print('The number of source points in obstacle：{}'.format(np.sum(shot * pre_obstacle)))

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    patchdict = np.load(orpath + 'patchdict.npy', allow_pickle=True)
    patchdict = patchdict.item()

    class namedict:
        def __init__(self):
            self.a = 1

    Parameter = namedict
    with open(orpath + 'parameter.txt', 'r') as f:
        for i in f.readlines():
            name = i.split(':')[0]
            value = i.split(':')[1].strip('\n')
            if not hasattr(Parameter, name):
                setattr(Parameter, name, float(value))
    start_receive_line_position = getattr(Parameter, 'start_receive_line_position')
    start_receive_point_position = getattr(Parameter, 'start_receive_point_position')
    dis_shot_line = getattr(Parameter, 'dis_shot_line')
    dis_shot_point = getattr(Parameter, 'dis_shot_point')
    dis_receive_line = getattr(Parameter, 'dis_receive_line')
    dis_receive_point = getattr(Parameter, 'dis_receive_point')
    shotbinsize = getattr(Parameter, 'shotbinsize')
    receivebinsize = getattr(Parameter, 'receivebinsize')

    if congruence:
        if set_region:
            Patchpath = Patchpath + 'limit/'
        else:
            Patchpath = Patchpath + 'unlimit/'
    else:
        Patchpath = Patchpath + 'full/'

    offshot, score_list, finalcmp, keep_same_ind = BackwardOptimization(shot, shotframe, receive, pre_obstacle,
                                                                        offset_obstacle, patchdict,
                                                                        start_receive_line_position,
                                                                        start_receive_point_position, dis_shot_line,
                                                                        dis_shot_point,
                                                                        dis_receive_line,
                                                                        dis_receive_point, min_dis_shot, shotbinsize,
                                                                        receivebinsize,
                                                                        para_list, punish_lambda,
                                                                        max_radius=max_radius, safe_radius=safe_radius,
                                                                        path=orpath, Patchpath=Patchpath,
                                                                        savepath=savedir,
                                                                        distance=distance,
                                                                        congruence=congruence, set_region=set_region,
                                                                        k=10,
                                                                        min_goal=min_goal,
                                                                        line_punishment=line_punishment,
                                                                        limit_dis_shot_point=limit_dis_shot_point,
                                                                        use_prior=use_prior,
                                                                        expand_boundary=expand_boundary, stop=stop)

    np.save(savedir + 'offshot.npy', offshot)
    np.save(savedir + 'offcmp.npy', finalcmp)
    plt.plot(score_list)
    plt.savefig(savedir + 'score.png')
    # plt.show()
    plt.close()

    vmax = np.max(finalcmp)
    cmap = 'rainbow'
    sns.heatmap(finalcmp, annot=False, vmax=vmax, vmin=0, cmap=cmap)
    plt.savefig(savedir + 'offset_cmp.png')
    # plt.show()
    plt.close()

    plt.figure(dpi=100)
    plt.scatter(np.where(offshot == 1)[0], np.where(offshot == 1)[1], s=0.3, color='red')
    plt.scatter(np.where(offset_obstacle == 1)[0], np.where(offset_obstacle == 1)[1], s=0.3, color='blue')
    plt.savefig(savedir + 'offset_observe_obstacle.png')
    # plt.show()
    plt.close()

    plt.figure(dpi=100)
    plt.scatter(np.where(offshot == 1)[0], np.where(offshot == 1)[1], s=0.3, color='red')
    plt.savefig(savedir + 'offset_observe.png')
    # plt.show()
    plt.close()

    print('The number of source points in obstacle：{}'.format(np.sum(offshot * offset_obstacle)))
    return keep_same_ind
