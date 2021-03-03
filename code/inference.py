import utils
import model
from model import MyModel
(train_img, train_lab),(test_img, test_lab) = data_loader("CIFAR10")
test_img = utils.data_preprocess(test_img)
test_lab = utils.one_hot_encoder(test_lab)

epochs = 10
resnet50_model = MyModel()
resnet50_model = resnet50_model.ResNet50()

def evaluate(test_im, test_lab):

    test_result = resnet50_model.evaluate(test_im, test_lab, verbose=0)
    return test_result

test_result = evaluate(test_img, test_lab)
print ("ResNet50 loss: ", test_result[0])
print ("ResNet50 accuracy: ", test_result[1])
