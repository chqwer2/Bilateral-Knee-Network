import argparse, os
from options import utils_option as option

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='', help='Path to option JSON file.')
parser.add_argument('--gpuid', type=str, default='0')

parser.add_argument('--dist', default=False)
parser.add_argument('--task', default="denoising")

parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')

parser.add_argument('--pretrain_dir', default=None)
parser.add_argument('--init_iter', type=int, default=0)


opt = option.parse(parser.parse_args().opt, is_train=True)
os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().gpuid
CUDA_VISIBLE_DEVICES = parser.parse_args().gpuid



