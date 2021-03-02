import tensorflow as tf
import san
from san import SAN
import config
import utils
import math

(train_img, train_lab),(test_img, test_lab) = utils.data_loader("CIFAR10")
train_count = len(train_img)
train_img = utils.data_preprocess(train_img)
# train_lab = utils.one_hot_encoder(train_lab)

X_train, X_val, y_train, y_val = utils.validation_data(train_img, train_lab)
train_generator, val_generator = utils.data_augmentation(X_train, y_train, X_val, y_val)
model = SAN()
model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
model.summary()

# define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adadelta()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step(images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)

# start training
for epoch in range(config.EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    step = 0
    for images, labels in train_generator:
        step += 1
        train_step(images, labels)
        print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                    config.EPOCHS,
                                                                                    step,
                                                                                    math.ceil(train_count / config.BATCH_SIZE),
                                                                                    train_loss.result(),
                                                                                    train_accuracy.result()))

    for valid_images, valid_labels in val_generator:
        valid_step(valid_images, valid_labels)

    print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
            "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                config.EPOCHS,
                                                                train_loss.result(),
                                                                train_accuracy.result(),
                                                                valid_loss.result(),
                                                                valid_accuracy.result()))

model.save_weights(filepath=config.save_model_dir, save_format='tf')