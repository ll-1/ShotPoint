import argparse

parser = argparse.ArgumentParser()
# the parameter of train the networks
parser.add_argument('--method', default='RL1005')
parser.add_argument('--viewsize', default=None)
parser.add_argument('--limit_len', default=None)
parser.add_argument('--learning_rate', default=2e-3, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_glismpses', default=4, type=int)
parser.add_argument('--n_process_block_iters', default=4, type=int)
parser.add_argument('--actor_lr', default=5e-4, type=float)
parser.add_argument('--critic_lr', default=5e-4, type=float)
parser.add_argument('--beta', default=0.001, type=float)
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--train_ratio', default=0.8, type=float)
parser.add_argument('--seed', default=50, type=int)
parser.add_argument('--max_epoch', default=300, type=int)
parser.add_argument('--max_grad_norm', default=2, type=float)
parser.add_argument('--lambda_r',default=16,type=int)

# define the dataset
# 设置偏移的工区
parser.add_argument('--name', default='example')
parser.add_argument('--testname', default='example')
# 设置坡度最大值，认为超过这个值的地方是障碍物
parser.add_argument('--limit_slope', default=20, type=float)
parser.add_argument('--max_relief', default=90, type=float)

parser.add_argument('--binsize', default=2, type=int)
parser.add_argument('--cmpbinsize', default=2, type=int)
parser.add_argument('--or_exist_elevation', default=False)
parser.add_argument('--off_exist_elevation', default=False)

# the dis of expand obstacle
parser.add_argument('--expand_obstacle', default=True)
parser.add_argument('--safe_radius', default=50, type=int)

# the max dis of shot points can offset.
# 设置炮点能够偏移的最大距离
parser.add_argument('--max_radius', default=240, type=int)

# the parameter of dynamic.
# 设置权重参数选择的优化目标
parser.add_argument('--min_goal', default='var')
parser.add_argument('--line_punishment', default=False)
parser.add_argument('--Is_AI', default=False)
# 设置优化算法中的目标
parser.add_argument('--var_goal', default=True)
parser.add_argument('--offset_goal', default=True)
parser.add_argument('--line_goal', default=True)
# 是否利用人事先偏移部分炮点的先验信息
parser.add_argument('--use_prior', default=False)

# 设置炮点间的距离，如果为True，则炮点间最小距离为炮距的一半；否则，炮点不重合即可。
parser.add_argument('--limit_dis_shot_point', default=True)
parser.add_argument('--min_dis_shot', default=40)
parser.add_argument('--distance', default='Euclidean')
parser.add_argument('--expand_boundary', default=10, type=int)
parser.add_argument('--stop',default=False)

# If run the main function of main.py to cover files.
parser.add_argument('--cover_file', default=False)

# CMP region.
parser.add_argument('--congruence', default=True)
parser.add_argument('--set_region', default=True)
# 存储施工区域内各个点的排列片，如果没有排列片，设置为True；如果有，则设为False.
parser.add_argument('--savecmp', default=False)
args = parser.parse_args()
