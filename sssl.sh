# -*- coding: utf-8 -*-
for video in "in_test2"
do
    mkdir /wiset/Output/SSSL/${video}
    mkdir /wiset/Output/SSSL/${video}/fixations 
        
    ffmpeg -i /wiset/Input/${video}/${video}.360  -map 0:6 /wiset/Input/${video}/${video}.wav
        
    octave -W /wiset/Localize/SSSL/mcsr/Main.m ${video}   # saliency.mat in /wiset/Output/SSSL/video_name
    python /wiset/Localize/SSSL/scripts/main.py --video_name ${video}   # pred.csv in /wiset/Output/SSSL/video_name

    python /wiset/Localize/sssl_fixation.py --video_name ${video}   # noise smoothing

    python /wiset/Localize/clsf.py --video_name ${video}    # event sound classification

    python /wiset/Localize/SSSL/scripts/fixmap2salmap.py --video_name ${video}   # fixation map in /wiset/Output/SSSL/video_name/fixations

    # ******************************************

    mkdir /wiset/Output/ViNet/${video}    
    
    python /wiset/Localize/ViNet/scripts/generate_result.py --video_name ${video}   # ViNet map in /wiset/Output/ViNet/video_name

    mkdir /wiset/Output/Fusion/${video}

    python /wiset/Localize/fusion.py --video_name ${video}   # fusion img in /wiset/Output/Fusion/video_name

    mkdir /wiset/Output/Final_Result/${video}
    mkdir /wiset/Output/Final_Result/${video}/Overlay
    
    python /wiset/Localize/overlay.py --video_name ${video}   # overlay img in /wiset/Output/Final_Result/video_name/Overlay
    
    ffmpeg -f image2 -r 30 -i /wiset/Output/Final_Result/${video}/Overlay/%d.jpg -vcodec mpeg4 ./tmp.mp4    # overlay img to video
    ffmpeg -i /wiset/Input/${video}/${video}.mp4 -vn ./tmp.m4a    # add audio
    ffmpeg -i ./tmp.mp4 -i ./tmp.m4a -c copy /wiset/Output/Final_Result/${video}/${video}.mp4 #-> result video in /wiset/Output/Final_Result/video_name  
        
    rm ./tmp.mp4 ./tmp.m4a

done
