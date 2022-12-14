import numpy as np
import os
import sys
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', default='in_test2', type=str, help='video path')
    return parser.parse_args()

def work(args):

    output_path = os.path.join('/wiset/Output/SSSL', args.video_name)
    fps = 30

    saliency_mat_path = os.path.join(output_path, '{}_saliency.mat'.format(args.video_name))
    #print("FPS:", fps, "Path:", saliency_mat_path)

    ch1_seconds, ch2_seconds, ch3_seconds = get_saliency_ratios(saliency_mat_path)    
    directional_saliencies = np.asarray([ch1_seconds, ch2_seconds, ch3_seconds]).T

    saliencies_as_unit_vector = np.apply_along_axis(to_unit_vector, 1, directional_saliencies)

    saliencies_as_UV_form = np.apply_along_axis(xyz2uv, 1, saliencies_as_unit_vector)
    uv_to_csv(saliencies_as_UV_form, os.path.join(output_path), fps)

if __name__ == '__main__':
    args = parse_args()
    work(args)