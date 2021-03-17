import numpy as np
import argparse
import os.path
import rosbag_pandas
import rosbag
import subprocess, yaml
from rosbag.bag import Bag
import sys


def return_fps(filename):
    info_dict = yaml.load(Bag(filename, 'r')._get_yaml_info())
    dur = info_dict['duration'] #video length in secs
    for d in info_dict['topics']:
        topic = d['topic']
        if d['topic'] == '/device_0/sensor_1/Color_0/image/data':
            n_frames = d['messages']
    fps = n_frames/dur
    print(f'{filename} \ndur = {dur} secs, fps={fps}')

#look for all bag files in provided path and return fps
if len(sys.argv) > 2:
    print('too many input arguments provided')
else:
    path = sys.argv[-1]
    files = os.listdir(path)
    for f in files:
        filename = os.path.join(path,f)
        if filename.endswith('.bag') is not True:
            continue
        try:
            return_fps(filename)
        except:
            print('file not found or something happened')
