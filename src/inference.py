import cv2
import argparse
import numpy as np
from collections import Counter
from training import c3d_model


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def get_data(self, filename):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_stack = np.array([])
        for n in range(0, nframe, self.depth):

            if  (nframe - n) < self.depth:
                break

            framearray = []
            for i in range(self.depth):

                ret, frame = cap.read()

                if ret:
                    frame = cv2.resize(frame, (self.height, self.width))
                    framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    print("Error reading frames")
                    break

            framearray = np.expand_dims(np.array(framearray), axis=0)
            frame_stack = np.vstack([frame_stack, framearray]) if frame_stack.size else framearray

        cap.release()

        return frame_stack


def load_video(video_path, vid3d):
    
    X = np.array([])
    filename = video_path
    print('filename :', filename)
    frame_stack = vid3d.get_data(filename)
    X = np.vstack([X, frame_stack]) if X.size else frame_stack

    print('No of stack in video :', X.shape[0])

    return X


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_rows", type=int, default=32)
    parser.add_argument("--img_cols", type=int, default=32)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--nb_classes", type=int, default=3)
    parser.add_argument("--inference_video", type=str,  default="inference/85600.mp4")
    parser.add_argument("--weights_data", type=str,  default="c3d_weights.h5")
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    activity = {0:'low', 1:'medium', 2:'high'}
    
    vid3d = Videoto3D(args.img_rows, args.img_cols, args.depth)
    x = load_video(args.inference_video, vid3d)
    x = x.reshape((x.shape[0], args.img_rows, args.img_cols, args.depth, args.channel))
    
    model = c3d_model(x, args.nb_classes)
    model.load_weights(args.weights_data)
    
    # Evaluate the deep learning model
    y_pred = model.predict(x, verbose=0)
    predicted = np.argmax(y_pred, axis=1)
    counter = Counter(predicted)
    print('Activity level of fishes in the video :', activity[max(counter)])
