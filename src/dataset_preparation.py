import cv2
import argparse
import numpy as np
from glob import glob
from random import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_all_file_names(input_path):

    names, labels = [], []

    file_names = glob(input_path+'/*.*')
    for file_name in file_names:
        print(file_name.split('\\'))
        if file_name.split('\\')[1].split('.')[0][-1] == '0':
            labels.append(0)
            names.append(file_name)
        elif file_name.split('\\')[1].split('.')[0][-1] == '1':
            labels.append(1)
            names.append(file_name)
        elif file_name.split('\\')[1].split('.')[0][-1] == '2':
            labels.append(2)
            names.append(file_name)

    c = list(zip(names, labels))
    shuffle(c)
    names, labels = zip(*c)

    return names, labels


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def get_data(self, filename, label):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('tot num of frames in the video :', nframe)

        frame_stack = np.array([])
        labels = []
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
            labels.append(label) 

        cap.release()

        return frame_stack, labels


def load_data(video_list, vid3d):
    X = np.array([])
    labels_list = []
    for idx, value in enumerate(video_list):
        # Display the progress
        if (idx % 100) == 0:
            print("process data %d/%d" % (idx, len(video_list)))
        filename = value
        label = int(value.split('\\')[1].split('.')[0][-1])
        print('filename :', filename, 'label :', label)
        frame_stack, labels = vid3d.get_data(filename, label)
        X = np.vstack([X, frame_stack]) if X.size else frame_stack
        labels_list.extend(labels)
        print('X.shape :', X.shape)

    return X, labels_list


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_rows", type=int, default=32)
    parser.add_argument("--img_cols", type=int, default=32)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--nb_classes", type=int, default=3)
    parser.add_argument("--input_path", type=str,  default="..\\train")
    parser.add_argument("--train_data", type=str,  default="../datasets/c3d_train_1.npz")    
    parser.add_argument("--test_data", type=str,  default="../datasets/c3d_test_1.npz")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    # Prepare training data
    names, labels = load_all_file_names(args.input_path)
    train, labels = list(names), list(labels)
    print('list of files :', train)

    vid3d = Videoto3D(args.img_rows, args.img_cols, args.depth)
    x, y = load_data(train, vid3d)
    x = x.reshape((x.shape[0], args.img_rows, args.img_cols, args.depth, args.channel))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    y_train, y_test = np.array(y_train), np.array(y_test)
    y_train = tf.keras.utils.to_categorical(y_train, args.nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, args.nb_classes)
    #print("Class values in the dataset are ... ", np.unique(y_train))
    print(' Dataset shape ;', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Save the dataset to be used for training later
    #np.savez(args.train_data, X=x_train, Y=y_train)
    #np.savez(args.test_data,  X=x_test,  Y=y_test)
