import os
#import cv2
#import sklearn
import random
import time
import numpy as np
from glob import glob
#from src.utils import extract_video3D, extract_video3D_optical_flow, extract_videos3D_frames_substraction


def extract_videos3D(video_input_file_path, height, width):
    video_frames = list()
    cap = cv2.VideoCapture(video_input_file_path)
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (width, height))
            video_frames.append(frame)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return video_frames

def extract_videos3D_optical_flow(video_input_file_path, height=720, width=1280):
    video_frames_optical_flow = []
    i = 0
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    file_name = video_input_file_path.split('\\')[1].split('.')[0]
    output_file = os.path.join('E:/RWF-2000/opt_flow/train', file_name+'.avi')

    cap = cv2.VideoCapture(video_input_file_path)
    ret1, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]
    vid_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))
    #frame1 = cv2.resize(frame1, (width, height))
    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    #print(output_file)
    if not cap.isOpened():
        print("Error opening video stream or file")
    
    while cap.isOpened():

        ret2, frame2 = cap.read()

        if ret2:

            frame2 = cv2.resize(frame2, (width, height))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            video_frames_optical_flow.append(bgr)
            vid_writer.write(bgr)
            cv2.waitKey(1)
        else:
            break

        i += 1
        prvs = next
    
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    return video_frames_optical_flow

def extract_videos3D_frames_substraction(video_input_file_path, height, width):
    video_frames = list()
    cap = cv2.VideoCapture(video_input_file_path)
    ret1, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (width, height))

    while cap.isOpened():

        ret2, frame2 = cap.read()
        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
            frame = frame1 - frame2
            video_frames.append(frame)
        else:
            break

        frame1 = frame2

    cap.release()
    cv2.destroyAllWindows()
    return video_frames

if __name__ == '__main__':

    # files = glob('E:/keypoint_videos/train/*.avi')
    # for file in files[0:2]:
    #     optical_flow = extract_videos3D_optical_flow(file)

    dir_name = 'E:/RWF-2000/train/'
    # Get list of all files only in the given directory
    # list_of_files = filter(lambda x: os.path.isfile(os.path.join(dir_name, x)),
    #                         os.listdir(dir_name))
    list_of_fi_files = glob(dir_name+'fi*.avi')
    list_of_no_files = glob(dir_name+'no*.avi')
    list_of_files = list_of_fi_files + list_of_no_files
    # Sort list of file names by size
    list_of_files = sorted(list_of_files, key=lambda x: os.stat(x).st_size, reverse=True)  
    # Iterate over sorted list of file names and
    # print them one by one along with size
    # for file_name in list_of_files[0:20]:
    #     file_path = os.path.join(dir_name, file_name)
    #     file_size = os.stat(file_path).st_size//1024
    #     print(file_size, ' -->', file_name)

    print(len(list_of_fi_files), len(list_of_no_files),  len(list_of_files))
