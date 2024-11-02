import cv2
import os
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline

from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from model import c3d_model_1, c3d_model_2

# Check GPU coinfiguration in Colab
print("Tensorflow version: ", tf.__version__)
print(tf.test.gpu_device_name())
##[1]
def load_all_file_names(input_path):

    names, labels = [], []
    random.seed(30)

    file_names = glob(input_path+'/*.mp4')
    lo_cnt, mi_cnt, hi_cnt = 0, 0, 0  
    for file_name in file_names:    
        if file_name.split('\\')[1].split('.')[0][-1] =='0':
            lo_cnt +=1
            labels.append(0)
            names.append(file_name)
        elif file_name.split('\\')[1].split('.')[0][-1] =='1':
            labels.append(1)
            names.append(file_name)
        elif file_name.split('\\')[1].split('.')[0][-1] =='2':
            labels.append(2)
            names.append(file_name)

    c = list(zip(names,labels))
    names, labels = zip(*c)

    return names, labels

input_path = "train"
#input_path = "optical_flow"
names, labels = load_all_file_names(input_path)
train, labels = list(names), list(labels)

print(train, labels)

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
        
        print('len(labels) :', len(labels))
            
        cap.release()

        return frame_stack, labels

    
def loaddata(video_list, vid3d):
    X = np.array([])
    labels_list = []
    for idx, value in enumerate(video_list[0:3]):
        # Display the progress
        if (idx % 100) == 0:
            print("process data %d/%d" % (idx, len(video_list)))
        filename = value
        label = int(value.split('.')[0][-1])
        print('filename :', filename, 'label :', label)
        frame_stack, labels = vid3d.get_data(filename, label)
        X = np.vstack([X, frame_stack]) if X.size else frame_stack
        labels_list.extend(labels)
        print(X.shape)
        print(len(labels_list))
    #return np.array(X).transpose((0, 2, 3, 1))
    #return np.array(X)
    return X, labels_list

class Args:
    batch = 128
    epoch = 50
    nclass = 3 # 11 action categories
    depth = 10
    rows = 32
    cols = 32
    skip = True # Skip: randomly extract frames; otherwise, extract first few frames

param_setting = Args()
img_rows = param_setting.rows
img_cols = param_setting.cols
frames = param_setting.depth
channel = 1
vid3d = Videoto3D(img_rows, img_cols, frames)
nb_classes = param_setting.nclass

# Prepare training data
x, y = loaddata(train, vid3d)
x = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
#c = list(zip(x, y))
#shuffle(c)
#x, y = zip(*c)
#y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

y_train, y_test = np.array(y_train), np.array(y_test)
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Define deep learning model
# This is simplified C3D model
c3d_model = Sequential()
c3d_model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(X_train.shape[1:]), padding='same'))
c3d_model.add(Activation('relu'))
c3d_model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
c3d_model.add(Activation('relu'))
c3d_model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
c3d_model.add(Dropout(0.2))

c3d_model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
c3d_model.add(Activation('relu'))
c3d_model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
c3d_model.add(Activation('relu'))
c3d_model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
c3d_model.add(Dropout(0.2))

c3d_model.add(Flatten(name='flatten_feature'))
c3d_model.add(Dense(512, activation='relu'))
c3d_model.add(Dropout(0.2))
c3d_model.add(Dense(nb_classes, activation='sigmoid'))

c3d_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
#c3d_model.summary()

# You can uncomment if you want to re-build the model; otherwise, load them from the data files

history = c3d_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=param_setting.batch, epochs=param_setting.epoch, verbose=1, shuffle=True)

#c3d_model.save_weights("c3d_opt_flow_RWF_2000.h5")

# Load the pre-trained model
# c3d_model.load_weights("c3d_RWF_2000.h5")

# Evaluate the deep learning model
y_pred = c3d_model.predict(x_test, verbose=0)
print("Confusion matrix")
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('Accuracy Score :', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('c3d_rgb_accuracy.eps', format='eps', dpi=1000)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('c3d_rgb_loss.eps', format='eps', dpi=1000)
plt.show()



