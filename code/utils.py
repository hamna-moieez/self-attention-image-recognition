import tensorflow as tf
from sklearn.model_selection import train_test_split

batch_size = 64
def data_loader(dataset):
    
    if dataset.startswith('MNIST'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset.startswith('CIFAR10'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    return (x_train, y_train), (x_test, y_test)

# (train_img, train_lab),(test_img, test_lab) = data_loader("MNIST")
# print(train_img.shape)

def one_hot_encoder(labels):

    ### One hot encoding for labels 

    lab_categorical = tf.keras.utils.to_categorical(
        labels, num_classes=10, dtype='uint8')

    return lab_categorical


def data_preprocess(images):
    return images/255.0

def validation_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=42)
    
    return X_train, X_val, y_train, y_val

# X_train, X_val, y_train, y_val = validation_data(train_img, train_lab)

# print(X_train.shape)

def data_augmentation(train_im, train_lab, valid_im, valid_lab):
    train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                width_shift_range=0.1, 
                                                                height_shift_range = 0.1, 
                                                                horizontal_flip=True)
 
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=batch_size) # train_lab is categorical 
    valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size) # so as valid_lab
    return train_set_conv, valid_set_conv