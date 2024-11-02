import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
print("Tensorflow Version :", tf.__version__)
print("CPU Physical Devices :", tf.config.list_physical_devices('CPU'))
print("GPU Physical Devices :", tf.config.list_physical_devices('GPU'))

def c3d_model(input_size, nb_classes):
    
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(input_size.shape[1:]), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten(name='flatten_feature'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    return model


def plot_history(history):

    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('c3d_accuracy.eps', format='eps', dpi=1000)
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'][1:])
    plt.plot(history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('c3d_loss.eps', format='eps', dpi=1000)
    plt.show()

    return

def load_datasets(train, test):
    with np.load("c3d_train.npz") as npzfile:
        x_train = npzfile["X"]
        y_train = npzfile["Y"]
    
    with np.load("c3d_test.npz") as npzfile:
        X_test = npzfile["X"]
        y_test = npzfile["Y"]
    
    # Split the training data further into training and validation set for training
    # The test data (an unseen data to the training) will be used for prediction
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.20, random_state = 0)
    
    print("Training data", X_train.shape, y_train.shape)
    print("Validation data", X_val.shape, y_val.shape)
    print("Test data", X_test.shape, y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--nb_classes", type=int, default=3)
    parser.add_argument("--train_data", type=str,  default="c3d_train.npz")
    parser.add_argument("--test_data", type=str,  default="c3d_test.npz")
    parser.add_argument("--weights_data", type=str,  default="c3d_weights.h5")
    
    return parser.parse_args()


if __name__ == '__main__':
    
    print('***** If GPU is used for training, at least 8 GB of GPU memory is recommended \n'
          '***** Alternatively you can turned off GPU and only use CPU for training, but it will be very slow')
    
    args = parse_args()
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_datasets(args.train_data, args.test_data)
    
    model = c3d_model(X_train, args.nb_classes)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=args.batch, epochs=args.epoch, verbose=1, shuffle=True)
    
    # Evaluate the deep learning model
    y_pred = model.predict(X_test, verbose=0)
    print("Confusion matrix")
    print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    print('Accuracy Score :', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    
    plot_history(history.history)
    