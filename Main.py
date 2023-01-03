import copy
import math
import os.path
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('./')
from method.Optimization.LimitClusterDynamicProgramming import LimitClusterBackwardOptimizationmain
from Config import args
import sys


def onemain(args):
    name = args.name
    testname = args.testname
    max_dem = args.limit_slope
    max_relief = args.max_relief
    binsize = args.binsize
    cmpbinsize = args.cmpbinsize
    max_radius = args.max_radius
    viewsize = args.viewsize
    min_goal = args.min_goal
    safe_radius = args.safe_radius
    line_punishment = args.line_punishment
    congruence = args.congruence
    set_region = args.set_region
    savecmp = args.savecmp
    cover_file = args.cover_file
    method = args.method
    expand_obstacle = args.expand_obstacle
    patchpath = '/home/longli/PycharmProjects/GEOSHOT/shot0718PatchCMP/' + str(name) + '/'
    savetime = str(datetime.datetime.now().year) + str(datetime.datetime.now().month) + str(
        datetime.datetime.now().day) + str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
    savetime = savetime + '_limit_' + str(args.limit_dis_shot_point)
    savetime = savetime + '_' + str(args.min_goal)
    para_list = []
    if args.var_goal and args.offset_goal and args.line_goal:
        savetime = savetime + '_var_offset_line'
        a0_list = np.exp(np.linspace(-4, np.log(1), 20))
        a1_list = np.exp(np.linspace(-4, np.log(1), 10))
        for a0 in a0_list:
            for a1 in a1_list:
                if a0 + a1 <= 1:
                    para_list.append([a0, a1])
    elif args.var_goal and args.offset_goal and not args.line_goal:
        savetime = savetime + '_var_offset'
        a0_list = np.exp(np.linspace(-4, np.log(1), 20))
        for a0 in a0_list:
            para_list.append([a0, 1 - a0])
    elif args.var_goal and not args.offset_goal and args.line_goal:
        savetime = savetime + '_var_line'
        a0_list = np.exp(np.linspace(-4, np.log(1), 20))
        for a0 in a0_list:
            para_list.append([a0, 0])
    elif not args.var_goal and args.offset_goal and args.line_goal:
        savetime = savetime + '_offset_line'
        a1_list = np.exp(np.linspace(-4, np.log(1), 20))
        for a1 in a1_list:
            para_list.append([0, a1])
    elif args.var_goal and not args.offset_goal and not args.line_goal:
        savetime = savetime + '_var_'
        para_list.append([1, 0])
    elif not args.var_goal and args.offset_goal and not args.line_goal:
        savetime = savetime + '_offset'
        para_list.append([0, 1])
    elif not args.var_goal and not args.offset_goal and args.line_goal:
        savetime = savetime + '_line'
        para_list.append([0, 0])

    if args.line_punishment:
        savetime = savetime + '_punishment'
    punish_lambda = np.linspace(1, 100, 5)

    basesavedir = './saveresult/' + os.path.basename(sys.argv[0]).split('.')[0] + '/' + name+ '/obstacle' + str(safe_radius)
    savedir = basesavedir + '/' + str(method) + '/' + savetime + '/'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    argsDict = args.__dict__
    f = open(savedir + 'config.txt', 'w')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.close()

    basetestresultpath = './data/' + os.path.basename(sys.argv[0]).split('.')[0] + '/' + testname
    basetestorpath = basetestresultpath + '/obstacle' + str(safe_radius)
    basetestoffpath = basetestresultpath + '/obstacle' + str(safe_radius)
    basetestsavedir = './saveresult/' + os.path.basename(sys.argv[0]).split('.')[0] + '/' + testname+ '/obstacle' + str(safe_radius)
    testorpath = basetestorpath + '/ordinary/'
    testoffpath = basetestoffpath + '/offset/'
    testsavedir = basetestsavedir+  '/' + str(method) + '/' + savetime + '/'

    eval(method + 'main')(testorpath, testoffpath, patchpath, max_radius=max_radius, safe_radius=safe_radius,
                          congruence=congruence,
                          set_region=set_region,
                          min_goal=min_goal,
                          savedir=savedir, line_punishment=line_punishment,
                          para_list=para_list, punish_lambda=punish_lambda,
                          limit_dis_shot_point=args.limit_dis_shot_point, min_dis_shot=args.min_dis_shot,
                          use_prior=args.use_prior,
                          distance=args.distance, expand_boundary=args.expand_boundary, stop=args.stop)

    # Quality Control
    cmp_ratio_list = [0.7, 0.8, 0.9, 1]
    for cmp_ratio in cmp_ratio_list:
        obstacle = np.load(testorpath + 'obstacle.npy', allow_pickle=True)
        ori_cmp = np.load(testorpath + 'cmp.npy', allow_pickle=True)
        print('original var:{}'.format(np.var(ori_cmp)))

        ori_full_coverage_ratio = np.sum(ori_cmp[ori_cmp >= cmp_ratio * np.max(ori_cmp)]) / np.sum(ori_cmp)
        print('original ratio:{}'.format(ori_full_coverage_ratio))
        pred_cmp=0
        pred_full_coverage_ratio=0
        MAE_ori_pred=0
        MSE_ori_pred=0
        if os.path.exists(testoffpath + 'cmp.npy'):
            off_cmp = np.load(testoffpath + 'cmp.npy', allow_pickle=True)
            print('offset var:{}'.format(np.var(off_cmp)))

            off_full_coverage_ratio = np.sum(off_cmp[off_cmp >= cmp_ratio * np.max(ori_cmp)]) / np.sum(ori_cmp)
            print('offset ratio:{}'.format(off_full_coverage_ratio))

            MAE_ori_offset = np.mean(np.abs(ori_cmp - off_cmp))
            MSE_ori_offset = np.mean((ori_cmp - off_cmp) ** 2)

        if os.path.exists(testsavedir + 'offcmp.npy'):
            pred_cmp = np.load(testsavedir + 'offcmp.npy', allow_pickle=True)
            print('pred var:{}'.format(np.var(pred_cmp)))

            pred_full_coverage_ratio = np.sum(pred_cmp[pred_cmp >= cmp_ratio * np.max(ori_cmp)]) / np.sum(ori_cmp)
            print('pred ratio:{}'.format(pred_full_coverage_ratio))

            MAE_ori_pred = np.mean(np.abs(ori_cmp - pred_cmp))
            MSE_ori_pred = np.mean((ori_cmp - pred_cmp) ** 2)

        f = open(testsavedir + 'result.txt', 'a')
        f.writelines('cmp ratio ：' + str(cmp_ratio) + '\n')
        f.writelines('original ratio : ' + str(ori_full_coverage_ratio) + '\n')
        f.writelines('original var : ' + str(np.var(ori_cmp)) + '\n')
        f.writelines('pred ratio : ' + str(pred_full_coverage_ratio) + '\n')
        f.writelines('pred var : ' + str(np.var(pred_cmp)) + '\n')
        f.writelines('MAE between ori and pred : ' + str(MAE_ori_pred) + '\n')
        f.writelines('MSE between ori and pred : ' + str(MSE_ori_pred) + '\n')
        if os.path.exists(testoffpath + 'cmp.npy'):
            f.writelines('offset ratio : ' + str(off_full_coverage_ratio) + '\n')
            f.writelines('offset var : ' + str(np.var(off_cmp)) + '\n')
            f.writelines('MAE between ori and offset : ' + str(MAE_ori_offset) + '\n')
            f.writelines('MSE between ori and offset : ' + str(MSE_ori_offset) + '\n')
        f.writelines('\n')
        f.writelines('\n')
        f.close()

    ori_shot = np.load(testorpath + 'regularshotarray.npy', allow_pickle=True)
    pred_shot = np.load(testsavedir + 'offshot.npy', allow_pickle=True)
    print('the number of shot points in obstacle:{}'.format(np.sum(pred_shot * obstacle)))

    if os.path.exists(testoffpath + 'regularshotarray.npy'):
        offset_shot = np.load(testoffpath + 'regularshotarray.npy', allow_pickle=True)
        print('the relation of offset and pred:{}'.format(np.sum(pred_shot * offset_shot)))

    print('original shot point number:{}'.format(np.sum(ori_shot)))
    print('pred shot point number:{}'.format(len(np.where(pred_shot != 0)[0])))

    f = open(testsavedir + 'result.txt', 'a')
    f.writelines('Original the number of shot points in obstacle : ' + str(np.sum(ori_shot * obstacle)) + '\n')
    f.writelines('Pred the number of shot points in obstacle : ' + str(np.sum(pred_shot * obstacle)) + '\n')
    f.writelines('original shot point number : ' + str(np.sum(ori_shot)) + '\n')
    f.writelines('pred shot point number : ' + str(len(np.where(pred_shot != 0)[0])) + '\n')
    f.writelines('\n')
    f.writelines('\n')
    f.close()


def multimain(args):
    functionlist = ['LimitClusterBackwardOptimization']
    punishlist = [False]
    limit_dis = [True]
    min_goal = ['var']  # 'var' , 'line_dis', 'offset_dis']
    var_goal = [False]
    offset_goal = [False]
    line_goal = [True]
    for a in functionlist:
        for b in punishlist:
            for l in limit_dis:
                for s in min_goal:
                    for c in var_goal:
                        for d in offset_goal:
                            for e in line_goal:
                                args.method = a
                                args.line_punishment = b
                                args.var_goal = c
                                args.offset_goal = d
                                args.line_goal = e
                                args.min_goal = s
                                args.limit_dis_shot_point = l
                                if c or d or e:
                                    onemain(args)

if __name__ == '__main__':
    multimain(args)


