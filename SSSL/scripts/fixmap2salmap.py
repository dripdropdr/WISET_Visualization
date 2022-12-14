import argparse
import glob
import os
import pandas as pd
import cv2
from vaODV import vaODV
from saliency_estimate import generate_saliencymap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', default='in_test2', type=str, help='video path')
    # parser.add_argument("--input", "-i", type=str, required=True, help="Input dataset_public folder location")            
    # parser.add_argument("--resolution", "-r", type=str, required=True, help="Resolution size of each ODV, Height x Width")
    return parser.parse_args()

def run(args):
    # height and width of a given ODV
    input_shape = [606,1080,3]

    # for st in SOUND_TYPE:
    va_odv = vaODV(vid_path=os.path.join('/wiset/Input', args.video_name), pred_path=os.path.join('/wiset/Output/SSSL', args.video_name), odv_shape=input_shape)
    
    # for odv_count, odv in enumerate(va_odv.odv_list):
    odv_name = args.video_name
        # va_odv.display_status(odv_count, odv_name)
    fixation_maps = va_odv.generate_fixations(odv_name)
            

if __name__ == "__main__":
                       
    args = parse_args()
    run(args)