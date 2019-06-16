#!/usr/local/bin/python3
"""
image.py -- Convolutional Neural Network trainer for self-driving cars.
Copyright (C) 2019  Alexis Nootens & Armen Hagopian

Licensed under the EUPL 1.2
Read license.txt for more details.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


class TrackGenerator(Sequence):
    """
    This class allows to combine the center, left, and right image inside
    the same training batch. It also allows to create several workers to
    multithread the training process.
    """

    def __init__(self, dataframe, preprocess, batch_size):
        self.dataframe = dataframe
        self.preprocess = preprocess
        self.batch_size = batch_size

        self.datagen = ImageDataGenerator(
            fill_mode='reflect',
            width_shift_range=0.1,
            height_shift_range=0.1,
            data_format='channels_last',
            preprocessing_function=preprocess,
            dtype=np.float32
        )

        self.camera_cent = self.datagen.flow_from_dataframe(
            dataframe=dataframe, x_col='Center', y_col='Steering',
            target_size=(160, 320), class_mode='other', shuffle=True,
            batch_size=batch_size, interpolation='bilinear')
        self.camera_righ = self.datagen.flow_from_dataframe(
            dataframe=dataframe, x_col='Right', y_col='Steering',
            target_size=(160, 320), class_mode='other', shuffle=True,
            batch_size=batch_size, interpolation='bilinear')
        self.camera_left = self.datagen.flow_from_dataframe(
            dataframe=dataframe, x_col='Left', y_col='Steering',
            target_size=(160, 320), class_mode='other', shuffle=True,
            batch_size=batch_size, interpolation='bilinear')

    def __getitem__(self, idx):
        sample_cent = self.camera_cent.__getitem__(idx)
        sample_left = self.camera_left.__getitem__(idx)
        sample_righ = self.camera_righ.__getitem__(idx)
        camera_comb = np.vstack((sample_cent[0], sample_left[0]))
        camera_comb = np.vstack((camera_comb, sample_righ[0]))
        rotate_comb = np.hstack((sample_cent[1], sample_left[1]+.25))
        rotate_comb = np.hstack((rotate_comb, sample_righ[1]-0.25))
        return camera_comb, rotate_comb

    def __len__(self):
        return self.dataframe.shape[0] // self.batch_size


def create_model():
    """ Returns a compiled model """
    model = Sequential([
        Cropping2D(cropping=((60, 0), (0, 0)), input_shape=(160, 320, 3),
                   data_format='channels_last'),
        Conv2D(24, kernel_size=5, strides=2, activation='relu'),
        Dropout(0.2),
        Conv2D(36, kernel_size=5, strides=2, activation='relu'),
        Dropout(0.2),
        Conv2D(48, kernel_size=5, strides=2, activation='relu'),
        Dropout(0.2),
        Conv2D(64, kernel_size=3, strides=1, activation='relu'),
        Dropout(0.2),
        Conv2D(64, kernel_size=3, strides=1, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(1)
    ])
    optimizer = RMSprop(lr=1e-6)
    model.compile(optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


def train_model(model, train_frame, valid_frame):
    """
    Fit the model with the dataframe. If no model is provided then load 'model.h5'.
    The model is trained with batches of the inputs whilst augmenting the data
    (i.e. rotating the image from 0 to 5°). The RGB pixels from 0 to 255 are
    linearly translated to the range [-1.0;1.0].
    """
    def preprocess(image):
        """ Distribute the inputs between [-1.0;1.0] """
        return preprocess_input(image, mode='tf', data_format='channels_last')

    batch_size = 32
    early_stop = EarlyStopping(monitor='mean_squared_error', patience=20,
                               mode='min', restore_best_weights=True)
    csv_logger = CSVLogger('model_history_log.csv')
    images_gen = TrackGenerator(train_frame, preprocess, batch_size)
    valids_gen = TrackGenerator(valid_frame, preprocess, batch_size)
    model.fit_generator(generator=images_gen, epochs=15,
                        validation_steps=batch_size,
                        validation_data=valids_gen,
                        workers=20, max_queue_size=500,
                        callbacks=[early_stop, csv_logger])
    model.save('model.h5')
    print("[INFO] model saved")


def read_driving_log(filename, one_of_five=False):
    """ Returns a dataframe with only the images paths and the steering values """
    column_names = ['Center', 'Left', 'Right', 'Steering',
                    'Accelerator', 'Brake', 'Speed']
    dataset = pd.read_csv(filename, names=column_names)
    dataset = dataset.drop(columns=['Accelerator', 'Brake', 'Speed'])
    dataset['Steering'] = dataset['Steering'].astype(np.float32)
    if one_of_five:
        to_take = range(0, dataset.shape[0], 5)
        dataset = dataset.take(to_take)
    return dataset


def plot_history(hist):
    """ Plot the MAE and MSE progress during training with matplotlib. """
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Steering angle]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Steering angle$^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Train a self-driving car.')
    PARSER.add_argument('-c', action='store_true', help='Create a new model')
    ARGS = PARSER.parse_args()

    MODEL = create_model() if ARGS.c else load_model('model.h5')
    DRV_LOG = read_driving_log('driving_log.csv')
    VAL_LOG = read_driving_log('validation_log.csv')
    train_model(MODEL, train_frame=DRV_LOG, valid_frame=VAL_LOG)

    HISTORY = pd.read_csv('model_history_log.csv')
    plot_history(HISTORY)
