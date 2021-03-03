import tensorflow as tf
import utils
import model
from model import MyModel

epochs = 150
batch_size = 64


def compile_network(model):
    model.compile(optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model.summary()


def train(train_dg, val_dg, net):
    lrdecay = tf.keras.callbacks.LearningRateScheduler(model.lrdecay) # learning rate decay  
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    history = net.fit(
        train_dg,
        validation_data=val_dg,
        epochs=epochs,
        callbacks=[lrdecay, early_stop]
        )
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    return train_acc, train_loss, val_acc, val_loss

def plot(acc, val_acc, loss, val_loss):
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
(train_img, train_lab),(test_img, test_lab) = utils.data_loader("CIFAR10")
train_img = utils.data_preprocess(train_img)
train_lab = utils.one_hot_encoder(train_lab)

X_train, X_val, y_train, y_val = utils.validation_data(train_img, train_lab)
train_generator, val_generator = utils.data_augmentation(X_train, y_train, X_val, y_val)

resnet50_model = MyModel()
resnet50_model = resnet50_model.resnet50(X_train)
summary = compile_network(resnet50_model)

train_acc, train_loss, val_acc, val_loss = train(train_generator, val_generator, resnet50_model)
print(train_acc, train_loss, val_acc, val_loss)

model.save_model(resnet50_model, "../models/ResNet50.h5")
