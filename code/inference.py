import utils
import model

(train_img, train_lab),(test_img, test_lab) = data_loader("MNIST")

epochs = 10
model = Model()
model = model.ResNet50()

def evaluate(test_im, test_lab):

    test_result = resnet50_model.evaluate(test_im, test_lab_categorical, verbose=0)
    return test_result

test_result = evaluate(test_img, test_lab)
print ("ResNet50 loss: ", test_result[0])
print ("ResNet50 accuracy: ", test_result[1])
