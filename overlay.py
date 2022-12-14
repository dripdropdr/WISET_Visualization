# -*- coding: utf-8 -*-
# 알파 블렌딩 (blending_alpha.py)

import os
import cv2
import numpy as np
import natsort
import argparse
from tqdm import tqdm
import glob

alpha = 0.5 # 합성에 사용할 알파 값


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, help='name of data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    frame_dir = os.path.join('/wiset/Input', args.video_name, 'frame_image')
    map_dir = os.path.join('/wiset/Output/Fusion', args.video_name, 'itti') 
    out_dir = os.path.join('/wiset/Output/Final_Result', args.video_name, 'Overlay')
    
    map_list = natsort.natsorted(os.listdir(os.path.join(map_dir)))

    if '.ipynb_checkpoints' in map_list:
        map_list.remove('.ipynb_checkpoints')
    
    for idx in tqdm(range(len(map_list)), desc="Mapping"):

        frame_img = cv2.imread(frame_dir + '/{0:04d}.jpg'.format(idx+1))
        fusion_img = cv2.imread(map_dir + '/{:04d}.png'.format(idx))
        
        frame_img = cv2.resize(frame_img, (1080, 606))
        
        blended = frame_img * alpha + fusion_img * (1-alpha)
        blended = blended.astype(np.uint8)

        respath = out_dir
        if not os.path.exists(respath):
            os.mkdir(respath)

        cv2.imwrite(respath + '/{}.jpg'.format(idx), blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
