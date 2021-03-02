import tensorflow as tf 
from config import NUM_CLASSES

def conv_1x1(out_dim, stride=1):
    return tf.keras.layers.Conv2D(filters=out_dim,
                            kernel_size=(1,1),
                            strides=stride,
                            padding="same",
                            use_bias=False) 

class SAN(tf.keras.Model):
    def __init__(self, sa_type=None, block=None, layers=None, kernels=None):
        super(SAN, self).__init__()
        channel = 64
        self.inital_conv = conv_1x1(channel)
        self.initial_bn = tf.keras.layers.BatchNormalization()
        self.conv0 = conv_1x1(channel)
        self.bn0 = tf.keras.layers.BatchNormalization()
        # self.layer0 = self.make_layer(sa_type, block, channel, layers[0], kernels=[0])
        
        channel *= 4
        self.conv1 = conv_1x1(channel)
        self.bn1 = tf.keras.layers.BatchNormalization()
        # self.layer1 = self.make_layer(sa_type, block, channel, layers[1], kernels=[1])

        channel *= 2
        self.conv2 = conv_1x1(channel)
        self.bn2 = tf.keras.layers.BatchNormalization()
        # self.layer2 = self.make_layer(sa_type, block, channel, layers[2], kernels=[2])

        channel *= 2
        self.conv3 = conv_1x1(channel)
        self.bn3 = tf.keras.layers.BatchNormalization()
        # self.layer3 = self.make_layer(sa_type, block, channel, layers[3], kernels=[3])

        channel *= 2
        self.conv4 = conv_1x1(channel)
        self.bn4 = tf.keras.layers.BatchNormalization()
        # self.layer4 = self.make_layer(sa_type, block, channel, layers[4], kernels=[4])

        # self.relu = tf.nn.relu()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(1,1))
        self.fc = tf.keras.layers.Dense(NUM_CLASSES)
    
    def make_layer(self, sa_type, block, planes, num_blocks, kernel_size=7, strides=1):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, strides))
        return tf.keras.Sequential(*layers)

    def call(self, input):
        x = tf.nn.relu(self.initial_bn(self.inital_conv(input)))
        x = tf.nn.relu(self.bn0(self.conv0(self.maxpool(x))))
        x = tf.nn.relu(self.bn1(self.conv1(self.maxpool(x))))
        x = tf.nn.relu(self.bn2(self.conv2(self.maxpool(x))))
        x = tf.nn.relu(self.bn3(self.conv3(self.maxpool(x))))
        x = tf.nn.relu(self.bn4(self.conv4(self.maxpool(x))))

        # x = self.relu(self.bn0(self.layer0(self.conv0(self.maxpool(x)))))
        # x = self.relu(self.bn1(self.layer1(self.conv1(self.maxpool(x)))))
        # x = self.relu(self.bn2(self.layer2(self.conv2(self.maxpool(x)))))
        # x = self.relu(self.bn3(self.layer3(self.conv3(self.maxpool(x)))))
        # x = self.relu(self.bn4(self.layer4(self.conv4(self.maxpool(x)))))
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def san(sa_type, layers, kernel, num_classes):
    return model 

