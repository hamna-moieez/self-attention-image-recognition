import tensorflow as tf
import numpy as np
from glob import glob
import cv2
import os
import random 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

def data_loader(dataset):
    if dataset.startswith('MNIST'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset.startswith('CIFAR10'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def one_hot_encoder(labels):
    lab_categorical = tf.keras.utils.to_categorical(
        labels, num_classes=config.NUM_CLASSES, dtype='uint8')
    return lab_categorical

def data_preprocess(images):
    return images/255.0

def validation_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=42)
    
    return X_train, X_val, y_train, y_val

def data_augmentation(train_im, train_lab, valid_im, valid_lab):
    train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                width_shift_range=0.1, 
                                                                height_shift_range = 0.1, 
                                                                horizontal_flip=True)
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=config.BATCH_SIZE) # train_lab is categorical 
    valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=config.BATCH_SIZE) # so as valid_lab
    return train_set_conv, valid_set_conv

def read_train_test_data(directory):
    train_img, train_lab = [], []
    test_img, test_lab = [], []
    for img_file in tqdm(glob(directory + "/train/*/*.JPEG")):
        train_label = img_file.split("/")[-2]
        img = cv2.imread(img_file)
        img = cv2.resize(img, (224, 224))
        train_img.append(img)
        train_lab.append(config.lbl2id[train_label])
    # np.save('train_images_224.npy', train_img)
    # np.save('train_labels_224.npy', train_lab)
    for img_file in tqdm(glob(directory + "/test/*/*.JPEG")):
        img = cv2.imread(img_file)
        test_label = img_file.split("/")[-2]
        img = cv2.resize(img, (224, 224))
        test_img.append(img)
        test_lab.append(config.lbl2id[test_label])

    x_train = np.asarray(train_img)
    y_train = np.asarray(train_lab)
    x_val = np.asarray(test_img)
    y_val = np.asarray(test_lab)
    
    return x_train, y_train, x_val, y_val

def lrdecay(epoch):
    lr = 1e-3
    if epoch > 50: lr *= 0.5e-3
    elif epoch > 20: lr *= 1e-3
    elif epoch > 10: lr *= 1e-2
    elif epoch > 5: lr *= 1e-1
    return lr

def smooth_loss(output, target, eps=0.1):
    w = tf.scatter(tf.expand_dims(target, 1), tf.zeros_like(output))
    # w = tf.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = tf.keras.layers.Softmax(output, axis=1)
    loss = tf.math.reduce_mean(tf.math.reduce_sum((-w * log_prob), axis=1))
    return loss
