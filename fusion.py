import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', default='in_test2', type=str, help='video path')
    return parser.parse_args()

def N(image):
    from scipy.ndimage.filters import maximum_filter
    M = 8.
    image = cv2.convertScaleAbs(image, alpha=M/image.max(), beta=0.)
    w,h = image.shape
    maxima = maximum_filter(image, size=(w/10,h/1))
    maxima = (image == maxima)
    mnum = maxima.sum()
    maxima = np.multiply(maxima, image)
    mbar = float(maxima.sum()) / mnum
    return image * (M-mbar)**2

def normalize_1(s_map):
	norm_s_map = (s_map - np.min(s_map)) / (s_map.max() - s_map.min())
	return 2*norm_s_map -1 

def normalize_map(s_map):
	norm_s_map = (s_map - np.min(s_map)) / (s_map.max() - s_map.min())
	return norm_s_map    

def fuse(video_name):

    audio_path = os.path.join('/wiset/Output/SSSL', args.video_name ,'fixations')
    visual_path = os.path.join('/wiset/Output/ViNet', args.video_name)

    for frame in tqdm(range(min(len(os.listdir(audio_path)), len(os.listdir(visual_path)))), desc='fusion:'):

        pred_audio_saliency = cv2.imread('/wiset/Output/SSSL/{}/fixations/salmap_f_{}.png'.format(video_name, frame), 0)
        pred_vinet = cv2.imread('/wiset/Output/ViNet/{}/{:04d}.jpg'.format(video_name, frame+1), 0)
        pred_vinet = cv2.resize(pred_vinet, (1080, 606))

        # print('/wiset/Output/SSSL/{}/fixations/salmap_f_{}.png'.format(video_name, frame), '/wiset/Output/ViNet/{}/{:04d}.jpg'.format(video_name, frame+1))

        output_path = os.path.join('/wiset/Output/Fusion', args.video_name, 'itti')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        pred_itti = 0.6*N(pred_audio_saliency) + 0.4*N(pred_vinet)

        p = output_path + '/{:04d}.png'.format(frame)

        cv2.imwrite(p, pred_itti)


def run(args):
    fuse(args.video_name)

if __name__ == "__main__":              
    args = parse_args()
    run(args)