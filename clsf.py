# -*- coding: utf-8 -*-
from email import header
import os
import sys
import argparse
import csv
import pandas as pd
import numpy as np
import imageio
import cv2
from tqdm import tqdm
import natsort



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', default='out_test2', type=str, help='video path')
    parser.add_argument('--cfg', default='balanced_mobile_Audio', type=str, help='hts cfg path')
    return parser.parse_args()

def get_odvInfo(vid_pth, odv_name):
        vid = imageio.get_reader(os.path.join(vid_pth, odv_name + '.mp4'), 'ffmpeg')
        return vid.get_meta_data()

def hts_csv(hts_pth):

    with open(hts_pth, newline='') as csvfile:
        hts_csv = csv.reader(csvfile, delimiter=' ', quotechar='|')
        classify_res = [line[0].split(',')[:2] for line in hts_csv if line[0].split(',')[2] != '0' and line[0].split(',')[2] != '3' ]
    
    classify_res = list(map(lambda x: list(map(lambda x: int(float(x)*30), x)), classify_res))
    classify_res.sort()

    return classify_res


def run(args):
    hts_pth = os.path.join('/wiset/hts/Result', args.cfg ,'Final_'+args.video_name+'.csv')
    vid_pth = os.path.join("/wiset/Input", args.video_name)
    fix_pth = os.path.join('/wiset/Output/SSSL', args.video_name, 'fixations')

    _img = imageio.imread(os.path.join('/wiset/Output/SSSL', args.video_name, 'fixations','salmap_f_' + str(0) + '.png'))
    vid_info = get_odvInfo(vid_pth, args.video_name)

    # hts_csv = pd.read_csv(hts_pth, header=None)
    classify_res = hts_csv(hts_pth)

    # frame = sorted(list(map(lambda x: int(x.rstrip('.jpg')), os.listdir(os.path.join(vid_pth, 'frame_image')))))
    fixmap_list = natsort.natsorted(os.listdir(os.path.join(fix_pth)))
    
    print('HTS configuration', args.cfg)

    print('*******Class exist frames*******')
    print(classify_res)

    try:
        for t in classify_res:
            print('class exist:', t[0], 'to', t[1])
            for f in range(t[0], t[1]+1):
                fixmap_list[f] = 0
    except IndexError:
        print('*******SSSL fixation map end*******')

    # print('*******remove fixation map list*******')
    # print(fixmap_list)

    for fixmap in tqdm(fixmap_list):
        if fixmap == 0:
            pass
        else:
            img_pth = (os.path.join(fix_pth, fixmap))
            imageio.imwrite(img_pth, np.zeros((_img.shape[0], _img.shape[1])).astype(np.uint8))


if __name__ == "__main__":
    args = parse_args()
    run(args)
