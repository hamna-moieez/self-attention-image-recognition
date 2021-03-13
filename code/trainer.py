import tensorflow as tf
import san
from san import SAN
import config
import utils
import math
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = san.san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7])
model.build(input_shape=(config.BATCH_SIZE, config.channels, config.image_height, config.image_width))
model.summary()

train_img, train_lab, test_img, test_lab= utils.read_train_test_data("../dataset")
train_img = utils.data_preprocess(train_img)
train_lab = utils.one_hot_encoder(train_lab)
X_train, X_val, y_train, y_val = utils.validation_data(train_img, train_lab)
train_generator, val_generator = utils.data_augmentation(X_train, y_train, X_val, y_val)

# define loss and optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        predictions = tf.reshape(predictions, (-1,10))
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step(images, labels):
    predictions = model(images, training=False)
    predictions = tf.reshape(predictions, (-1,10))
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)

# start training
for epoch in range(config.EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    for step in range(len(train_generator)):
        images, labels = train_generator[step]
        # images = tf.image.resize(images, [224, 224])
        images = tf.transpose(images, [0, 3, 1, 2])
        train_step(images, labels)

        print("Epoch: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                config.EPOCHS,
                                                                train_loss.result(),
                                                                train_accuracy.result()))

    for val_step in range(len(val_generator)):
        valid_images, valid_labels = val_generator[val_step]
        # valid_images = tf.image.resize(valid_images, [224, 224])
        valid_images = valid_images.transpose(0, 3, 1, 2)
        valid_step(valid_images, valid_labels)
    
    print("<"+"-"*80+">")
    print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
            "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                config.EPOCHS,
                                                                train_loss.result(),
                                                                train_accuracy.result(),
                                                                valid_loss.result(),
                                                                valid_accuracy.result()))
    print("<"+"-"*80+">")

model.save_weights(filepath=config.save_model_dir, save_format='tf')
