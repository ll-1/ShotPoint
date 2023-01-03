import shutil

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# Calculate the difference between patch cmp and original cmp to ensure the patches are correct.
def diff_PatchCMP_CMP(shotarray, Patchpath, orpath, savepath, congruence, dis_shot_line, dis_receive_line,
                      start_receive_line_position, start_receive_point_position):
    init_cmparray = np.zeros_like(shotarray)
    x1, y1 = np.where(shotarray != 0)
    for ind in tqdm(range(len(x1))):
        x = x1[ind]
        y = y1[ind]
        for t in range(int(shotarray[x, y])):
            if congruence:
                if not os.path.exists(Patchpath + 'NotCoverage/' + str(x) + '_' + str(y) + '.npy'):
                    congruencex = int((x - start_receive_point_position) % dis_shot_line)
                    congruencey = int((y - start_receive_line_position) % dis_receive_line)
                    readpath = Patchpath + 'Coverage/' + str(congruencex) + '_' + str(congruencey) + '.npy'
                    direct_cmp = np.load(readpath, allow_pickle=True)
                    direct_cmp = direct_cmp.item()[(congruencex, congruencey)]
                    init_cmparray[int(x - direct_cmp[0]):int(x + direct_cmp[2]),
                    int(y - direct_cmp[1]): int(y + direct_cmp[3])] += direct_cmp[4]
                else:
                    direct_cmp = np.load(Patchpath + 'NotCoverage/' + str(x) + '_' + str(y) + '.npy', allow_pickle=True)
                    direct_cmp = direct_cmp.item()[(x, y)]
                    init_cmparray[int(x - direct_cmp[0]):int(x + direct_cmp[2]),
                    int(y - direct_cmp[1]): int(y + direct_cmp[3])] += direct_cmp[4]
            else:
                direct_cmp = np.load(Patchpath + 'EveryCMP/' + str(x) + '_' + str(y) + '.npy', allow_pickle=True)
                direct_cmp = direct_cmp.item()[(x, y)]
                init_cmparray[int(direct_cmp[0]):int(direct_cmp[2]), int(direct_cmp[1]):int(direct_cmp[3])] += \
                    direct_cmp[4]
    vmax = np.max(init_cmparray)
    cmap = 'rainbow'
    sns.heatmap(init_cmparray, annot=False, vmax=vmax, vmin=0, cmap=cmap)
    # plt.savefig(savedir + 'init_cmp.png')
    plt.show()
    plt.close()
    if orpath != None:
        cmp = np.load(orpath + 'cmp.npy', allow_pickle=True)
        print('The difference between two cmps:{}'.format(np.sum(np.abs(cmp - init_cmparray))))
    else:
        np.save(savepath + 'cmp.npy', init_cmparray)


# Calculate the difference between cmp which is defined by the receive points and original cmp to ensure the receive points are correct.
def cmp_array(shotarray,receivearray, relationdict, orpath, savepath, dis_receive_line, dis_receive_point):
    init_cmparray = np.zeros_like(shotarray)
    x, y = np.where(shotarray != 0)
    for ind in tqdm(range(len(x))):
        i, j = x[ind], y[ind]
        for t in range(int(shotarray[i, j])):
            receive_set = relationdict[(i, j)]
            for ss in np.arange(receive_set[0][0], receive_set[0][1], dis_receive_point):
                for tt in np.arange(receive_set[1][0], receive_set[1][1], dis_receive_line):
                    if receivearray[int(ss), int(tt)] == 1:
                        ind_x = int((ss + i) / 2)
                        ind_y = int((tt + j) / 2)
                        init_cmparray[ind_x, ind_y] += 1
    if orpath != None:
        cmp = np.load(orpath + 'cmp.npy', allow_pickle=True)
        print(np.sum(np.abs(cmp - init_cmparray)))
    else:
        np.save(savepath + 'cmp.npy', init_cmparray)
    return init_cmparray


def point_cmp_array(point, receive_set, dis_receive_line, dis_receive_point):
    [i, j] = point
    xx, yy = np.meshgrid(np.arange(receive_set[0][0], receive_set[0][1], dis_receive_point),
                         np.arange(receive_set[1][0], receive_set[1][1], dis_receive_line))
    xx = xx.flatten()
    yy = yy.flatten()
    xlist = []
    ylist = []
    for ind2 in range(len(xx)):
        ind_x = int((i + xx[ind2]) / 2)
        ind_y = int((j + yy[ind2]) / 2)
        xlist.append(ind_x)
        ylist.append(ind_y)
    a = sorted(set(xlist.copy()))
    b = sorted(set(ylist.copy()))
    n = a[-1] - a[0] + 1
    p = b[-1] - b[0] + 1
    pointcmparray = np.zeros((n, p))
    for ind in range(len(xlist)):
        pointcmparray[xlist[ind] - a[0], ylist[ind] - b[0]] += 1
    return [a[0], b[0], a[-1] + 1, b[-1] + 1, pointcmparray]


# Define the cmp patch of every point.
def PatchCMP(patchdict, receive_line_number, receive_point_number, dis_receive_line, dis_receive_point, patchpath):
    nlist = [i for (i, j) in patchdict.keys()]
    plist = [j for (i, j) in patchdict.keys()]
    n, p = sorted(set(nlist))[-1] + 1, sorted(set(plist))[-1] + 1
    patchpath = patchpath + 'full/'
    if not os.path.exists(patchpath):
        os.makedirs(patchpath)
    for i in tqdm(range(n)):
        for j in range(p):
            if (i, j) in patchdict.keys():
                list = point_cmp_array([i, j], patchdict[(i, j)], dis_receive_line,
                                       dis_receive_point)
                savedict = dict()
                savedict[(i, j)] = [list[0], list[1], list[2], list[3], list[4]]
                np.save(patchpath + str(i) + '_' + str(j) + '.npy', savedict)


# According to the congruence relation, define the cmp patch of every point.
def Congruence_PatchCMP(shotarray, patchdict, start_receive_line_position, start_receive_point_position,
                        receive_line_number,
                        receive_point_number, dis_shot_line, dis_receive_line, dis_receive_point,
                        congruence_patchpath, set_region=True,expand_boundary=0):
    if set_region:
        congruence_patchpath = congruence_patchpath + 'limit/'
    else:
        congruence_patchpath = congruence_patchpath + 'unlimit/'
    if not os.path.exists(congruence_patchpath):
        os.makedirs(congruence_patchpath)
        notcover_patchpath = congruence_patchpath + 'NotCoverage/'
        cover_patchpath = congruence_patchpath + 'Coverage/'
        os.makedirs(notcover_patchpath)
        os.makedirs(cover_patchpath)
    else:
        shutil.rmtree(congruence_patchpath)
        notcover_patchpath = congruence_patchpath + 'NotCoverage/'
        cover_patchpath = congruence_patchpath + 'Coverage/'
        os.makedirs(notcover_patchpath)
        os.makedirs(cover_patchpath)

    nlist = [i for (i, j) in patchdict.keys()]
    plist = [j for (i, j) in patchdict.keys()]
    n, p = sorted(set(nlist))[-1] + 1, sorted(set(plist))[-1] + 1

    Cover_patchcmpdict = dict()
    NotCover_patchcmpdict = dict()

    min_i = 0
    min_j = 0
    max_i = n
    max_j = p
    if set_region:
        min_i = max(sorted(set(np.where(shotarray != 0)[0]))[0]-expand_boundary,0)
        min_j = max(sorted(set(np.where(shotarray != 0)[1]))[0]-expand_boundary,0)
        max_i = min(sorted(set(np.where(shotarray != 0)[0]))[-1] + 1+expand_boundary,n)
        max_j = min(sorted(set(np.where(shotarray != 0)[1]))[-1] + 1+expand_boundary,p)

    for i in tqdm(range(min_i, max_i)):
        for j in range(min_j, max_j):
            if (i, j) in patchdict.keys():
                if (patchdict[(i, j)][0][1] - patchdict[(i, j)][0][0]) >= dis_receive_point * receive_point_number and (
                        patchdict[(i, j)][1][1] - patchdict[(i, j)][1][0]) >= dis_receive_line * receive_line_number:
                    congruence_i = int((i - start_receive_point_position) % dis_shot_line)
                    congruence_j = int((j - start_receive_line_position) % dis_receive_line)
                    if not os.path.exists(cover_patchpath + str(congruence_i) + '_' + str(congruence_j) + '.npy'):
                        if (congruence_i, congruence_j) not in Cover_patchcmpdict.keys():
                            list = point_cmp_array([i, j], patchdict[(i, j)], dis_receive_line,
                                                   dis_receive_point)
                            Cover_patchcmpdict[(congruence_i, congruence_j)] = 1
                            savedict = dict()
                            savedict[(congruence_i, congruence_j)] = [i - list[0], j - list[1], list[2] - i, list[3] - j,
                                                                      list[4]]
                            np.save(cover_patchpath + str(congruence_i) + '_' + str(congruence_j) + '.npy', savedict)
                else:
                    if not os.path.exists(notcover_patchpath + str(i) + '_' + str(j) + '.npy'):
                        list = point_cmp_array([i, j], patchdict[(i, j)], dis_receive_line,
                                               dis_receive_point)
                        savedict = dict()
                        savedict[(i, j)] = [i - list[0], j - list[1], list[2] - i, list[3] - j, list[4]]
                        np.save(notcover_patchpath + str(i) + '_' + str(j) + '.npy', savedict)


def GraphPatch(shotarray, receivearray, patchdict, dis_receive_line, dis_receive_point, epsilon, orpath, savepath,
               congruence, setregion):
    init_cmparray = cmp_array(shotarray, patchdict, orpath, savepath, dis_receive_line, dis_receive_point)
    vmax = np.max(init_cmparray)
    ratio = np.sum(init_cmparray == vmax) / (init_cmparray.size)
    print(ratio)
    cmap = 'rainbow'
    sns.heatmap(init_cmparray, annot=False, vmax=vmax, vmin=0, cmap=cmap)
    plt.show()
    plt.close()


    