import argparse

from models import *
from utils_lib.datasets import *
from utils_lib.utils import *
from utils_lib.util import *
from networks.network import First_Third_Net
from utils_lib import torch_utils
import visdom
from scipy.io import loadmat
from networks import *
import networks
import cv2
import numpy as np
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int((x.split(".")[0])))
    try:
        for i in range(len(files)):
            filename = pathIn + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            print(filename)
            # inserting the frames into an image array
            frame_array.append(img)
    except:
        print('Done')

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

fps = 25.0
# Save Frame place
pathIn = '/home/yangmingwen/first_third_person/demo_video42/'
# Save Video place
pathOut = '/home/yangmingwen/first_third_person/demo/video_42.mp4'
# ==============================
convert_frames_to_video(pathIn, pathOut, fps)