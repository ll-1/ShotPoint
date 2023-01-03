import math
import copy
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


def adapative_search_radius_neighbor(array, x, y, k, max_radius, min_i, max_i, min_j, max_j):
    k_nearest_points = []
    radius = 1
    while radius < max_radius:
        newxlist = [max(min_i, x - radius), min(x + radius + 1, max_i)]
        for newx in newxlist:
            if min_j <= y < max_j and array[newx, y] != 0 and not np.isnan(
                    array[newx, y]) and not np.isinf(array[newx, y]):
                if [newx, y] not in k_nearest_points:
                    k_nearest_points.append([newx, y])
        radius += 1
    if k_nearest_points == []:
        radius = 1
        while radius < max_radius:
            for newx in range(max(min_i, x - radius), min(x + radius + 1, max_i)):
                newy = y - (radius - abs(newx - x))
                if min_j <= y < max_j and array[newx, newy] != 0 and not np.isnan(
                        array[newx, newy]) and not np.isinf(array[newx, newy]):
                    if [newx, newy] not in k_nearest_points:
                        k_nearest_points.append([newx, newy])
                newy = y + (radius - abs(newx - x))
                if min_j <= y < max_j and array[newx, newy] != 0 and not np.isnan(
                        array[newx, newy]) and not np.isinf(array[newx, newy]):
                    if [newx, newy] not in k_nearest_points:
                        k_nearest_points.append([newx, newy])
            radius += 1
    return k_nearest_points


def adapative_search_nearest_neighbor(array, x, y, k, max_radius, min_i, max_i, min_j, max_j, shotbinsize,
                                      receivebinsize, distance):
    k_nearest_points = []
    max_receive_radius = int(max_radius / receivebinsize)
    max_shot_radius = int(max_radius / shotbinsize)
    radius = 1
    while len(k_nearest_points) < k:
        newxlist = [max(min_i, x - radius), min(x + radius, max_i - 1)]
        for newx in newxlist:
            if min_j <= y < max_j and array[newx, y] != 0 and not np.isnan(
                    array[newx, y]) and not np.isinf(array[newx, y]):
                if [newx, y] not in k_nearest_points:
                    k_nearest_points.append([newx, y])
        radius += 1
        if radius > max_receive_radius:
            break
    if distance == 'Euclidean':
        if k_nearest_points == []:  # or len(k_nearest_points) < k:
            radius = 1
            # while k_nearest_points == []:
            while len(k_nearest_points) < k:
                newxlist = [max(min_i, x - radius), min(x + radius, max_i - 1)]
                for newx in newxlist:
                    for newy in range(max(min_j, y - radius), min(max_j, y + radius + 1)):
                        if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                                array[newx, newy]) and not np.isinf(
                            array[newx, newy]) and min_i <= newx < max_i:
                            if [newx, newy] not in k_nearest_points:
                                k_nearest_points.append([newx, newy])
                newylist = [max(min_j, y - radius), min(y + radius, max_j - 1)]
                for newy in newylist:
                    for newx in range(max(min_i, x - radius), min(max_i, x + radius + 1)):
                        if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                                array[newx, newy]) and not np.isinf(array[newx, newy]) and min_i <= newx < max_i:
                            if [newx, newy] not in k_nearest_points:
                                k_nearest_points.append([newx, newy])
                radius += 1
                # if radius > min(max_shot_radius,max_shot_radius):
                #     break
    elif distance == 'Manhattan':
        if k_nearest_points == []:  # or len(k_nearest_points) < k:
            radius = 1
            # while k_nearest_points == []:
            while len(k_nearest_points) < k:
                newxlist = np.arange(max(min_i, x - radius), min(x + radius + 1, max_i))
                for newx in newxlist:
                    newy = y - (radius - abs(newx - x))
                    if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                            array[newx, newy]) and not np.isinf(
                        array[newx, newy]) and min_i <= newx < max_i:
                        if [newx, newy] not in k_nearest_points:
                            k_nearest_points.append([newx, newy])
                    newy = y + (radius - abs(newx - x))
                    if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                            array[newx, newy]) and not np.isinf(
                        array[newx, newy]) and min_i <= newx < max_i:
                        if [newx, newy] not in k_nearest_points:
                            k_nearest_points.append([newx, newy])
                radius += 1
                # if radius > min(max_shot_radius,max_shot_radius):
                #     break

    if len(k_nearest_points) > k:
        # select_points = random.sample(k_nearest_points, k)
        select_points = k_nearest_points[:k]
        return select_points
    else:
        return k_nearest_points


def search_no_point(shotarray, x, y, safedis, shotbinsize, receivebinsize):
    n, p = shotarray.shape
    disx = math.ceil(safedis / receivebinsize)
    disy = math.ceil(safedis / shotbinsize)
    for i in range(max(0, x - disx), min(n, x + disx + 1)):
        for j in range(max(0, y - disy), min(p, y + disy + 1)):
            if shotarray[i, j] != 0 and ((i - x) * receivebinsize) ** 2 + ((j - y) * shotbinsize) ** 2 < safedis ** 2:
                return False
    return True


def limit_adapative_search_nearest_neighbor(array, candidate_shot, x, y, k, max_radius, min_i, max_i, min_j, max_j,
                                            min_dis_shot, shotbinsize,
                                            receivebinsize, distance='Euclidean', stop=False):
    max_receive_radius = int(max_radius / receivebinsize)
    max_shot_radius = int(max_radius / shotbinsize)
    k_nearest_points = []
    radius = 1
    while len(k_nearest_points) < k:
        newxlist = [max(min_i, x - radius), min(x + radius, max_i - 1)]
        for newx in newxlist:
            if min_j <= y < max_j and array[newx, y] != 0 and not np.isnan(
                    array[newx, y]) and not np.isinf(array[newx, y]):
                if [newx, y] not in k_nearest_points and search_no_point(candidate_shot, newx, y, min_dis_shot,
                                                                         shotbinsize,
                                                                         receivebinsize):
                    k_nearest_points.append([newx, y])
        radius += 1
        if radius > max_receive_radius:
            break
    if distance == 'Euclidean':
        if k_nearest_points == []:  # or len(k_nearest_points) < k:
            radius = 1
            while k_nearest_points == []:
                # while len(k_nearest_points) < k:
                newxlist = [max(min_i, x - radius), min(x + radius, max_i - 1)]
                for newx in newxlist:
                    for newy in range(max(min_j, y - radius), min(max_j, y + radius + 1)):
                        if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                                array[newx, newy]) and not np.isinf(
                            array[newx, newy]) and min_i <= newx < max_i:
                            if [newx, newy] not in k_nearest_points and search_no_point(candidate_shot, newx, newy,
                                                                                        min_dis_shot,
                                                                                        shotbinsize, receivebinsize):
                                k_nearest_points.append([newx, newy])
                newylist = [max(min_j, y - radius), min(y + radius, max_j - 1)]
                for newy in newylist:
                    for newx in range(max(min_i, x - radius), min(max_i, x + radius + 1)):
                        if min_i <= newx < max_i and array[newx, newy] != 0 and not np.isnan(
                                array[newx, newy]) and not np.isinf(
                            array[newx, newy]) and min_i <= newx < max_i:
                            if [newx, newy] not in k_nearest_points and search_no_point(candidate_shot, newx, newy,
                                                                                        min_dis_shot,
                                                                                        shotbinsize, receivebinsize):
                                k_nearest_points.append([newx, newy])
                radius += 1
                if stop:
                    if radius > min(max_shot_radius, max_shot_radius):
                        break
    elif distance == 'Manhattan':
        if k_nearest_points == []:  # or len(k_nearest_points) < k:
            radius = 1
            while k_nearest_points == []:
                # while len(k_nearest_points) < k:
                newxlist = np.arange(max(min_i, x - radius), min(x + radius + 1, max_i))
                for newx in newxlist:
                    newy = y - (radius - abs(newx - x))
                    if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                            array[newx, newy]) and not np.isinf(
                        array[newx, newy]) and min_i <= newx < max_i:
                        if [newx, newy] not in k_nearest_points and search_no_point(candidate_shot, newx, newy,
                                                                                    min_dis_shot,
                                                                                    shotbinsize, receivebinsize):
                            k_nearest_points.append([newx, newy])
                    newy = y + (radius - abs(newx - x))
                    if min_j <= newy < max_j and array[newx, newy] != 0 and not np.isnan(
                            array[newx, newy]) and not np.isinf(
                        array[newx, newy]) and min_i <= newx < max_i:
                        if [newx, newy] not in k_nearest_points and search_no_point(candidate_shot, newx, newy,
                                                                                    min_dis_shot,
                                                                                    shotbinsize, receivebinsize):
                            k_nearest_points.append([newx, newy])
                radius += 1
                if stop:
                    if radius > min(max_shot_radius, max_shot_radius):
                        break
    else:
        raise 'Not found the distance.'
    if len(k_nearest_points) > k:
        # select_points = random.sample(k_nearest_points, k)
        select_points = k_nearest_points[:k]
        return select_points
    else:
        return k_nearest_points
