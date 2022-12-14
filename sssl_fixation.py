# -*- coding: utf-8 -*-
from audioop import avg
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
    parser.add_argument('--video_name', default='out_test1', type=str, help='video path')
    return parser.parse_args()


def run(args):
    predcsv = pd.read_csv(os.path.join('/wiset/Output/SSSL', args.video_name, 'pred.csv'))
    
    for f in tqdm(range(predcsv.shape[0]//1600)):
        mean1 = predcsv['2dmu'][(f*1600) : (f*1600)+1600].mean()
        mean2 = predcsv['2dmv'][(f*1600) : (f*1600)+1600].mean()
        
        # Frame normalization
        # First frame
        if f == 0:
            mean1 = (mean1 + predcsv['2dmu'][(f*1600)+1600 : (f*1600)+3200].mean() + predcsv['2dmu'][(f*1600)+3200 : (f*1600)+4800].mean())/3
            mean2 = (mean2 + predcsv['2dmv'][(f*1600)+1600 : (f*1600)+3200].mean() + predcsv['2dmv'][(f*1600)+3200 : (f*1600)+4800].mean())/3
        # Second frame
        elif f == 1:
            mean1 = (predcsv['2dmu'][(f*1600)-1600 : (f*1600)].mean() + mean1 + predcsv['2dmu'][(f*1600)+1600 : (f*1600)+3200].mean() + predcsv['2dmu'][(f*1600)+3200 : (f*1600)+4800].mean())/4
            mean2 = (predcsv['2dmv'][(f*1600)-1600 : (f*1600)].mean() + mean2 + predcsv['2dmv'][(f*1600)+1600 : (f*1600)+3200].mean() + predcsv['2dmv'][(f*1600)+3200 : (f*1600)+4800].mean())/4
        # Last frame
        elif f == predcsv.shape[0]//1600 - 1:
            mean1 = (predcsv['2dmu'][(f*1600)-3200 : (f*1600)-1600].mean() + predcsv['2dmu'][(f*1600)-1600 : (f*1600)].mean() + mean1)/3
            mean2 = (predcsv['2dmv'][(f*1600)-3200  :(f*1600)-1600].mean() + predcsv['2dmv'][(f*1600)-1600  :(f*1600)].mean() + mean2)/3
        # Last-1 frame
        elif f == predcsv.shape[0]//1600 - 2:
            mean1 = (predcsv['2dmu'][(f*1600)-3200 : (f*1600)-1600].mean() + predcsv['2dmu'][(f*1600)-1600 : (f*1600)].mean() + mean1 + predcsv['2dmu'][(f*1600)+1600 : (f*1600)+3200].mean())/4
            mean2 = (predcsv['2dmv'][(f*1600)-3200  :(f*1600)-1600].mean() + predcsv['2dmv'][(f*1600)-1600  :(f*1600)].mean() + mean2 + predcsv['2dmv'][(f*1600)+1600 : (f*1600)+3200].mean())/4
        # else
        else:
            mean1 = (predcsv['2dmu'][(f*1600)-3200 : (f*1600)-1600].mean() + predcsv['2dmu'][(f*1600)-1600 : (f*1600)].mean() + mean1 + predcsv['2dmu'][(f*1600)+1600 : (f*1600)+3200].mean() + predcsv['2dmu'][(f*1600)+3200 : (f*1600)+4800].mean())/5
            mean2 = (predcsv['2dmv'][(f*1600)-3200 : (f*1600)-1600].mean() + predcsv['2dmv'][(f*1600)-1600 : (f*1600)].mean() + mean1 + predcsv['2dmv'][(f*1600)+1600 : (f*1600)+3200].mean() + predcsv['2dmv'][(f*1600)+3200 : (f*1600)+4800].mean())/5

        # print(f, mean1, mean2)
        # put value in each sound frame
        for sf in range((f*1600), (f*1600)+1600):
            predcsv['2dmu'][sf] = mean1
            predcsv['2dmv'][sf] = mean2

    predcsv.to_csv(os.path.join('/wiset/Output/SSSL', args.video_name, '_pred.csv'), sep=',', na_rep='NaN', index=False)



if __name__ == "__main__":
    args = parse_args()
    run(args)
