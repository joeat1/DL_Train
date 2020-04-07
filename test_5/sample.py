import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def pre_data(images, labels, size=100):
    len_data = len(images)
    num_each = int(size//10)  #每个数字创造的样本数
    num_each_all = np.bincount(labels)
    index = labels.argsort() # 按照0-9顺序排序好，index为原始下标
    image_d = []
    cur_ind=0
    for i in range(10):
        image_d.extend(images[index[cur_ind:cur_ind+10]])
        cur_ind += num_each_all[i]
    image_d = np.array(image_d)
    return image_d

image_d = pre_data(x_test, y_test)

n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

def SeNetBlock(feature, reduction=4):
    channels = feature.shape[3] #得到feature的通道数量w
    avg_x = tf.keras.layers.GlobalAveragePooling2D()(feature) #先对feature的每个通道进行Global Average Pooling 得到通道描述子（Squeeze）
    avg_x = tf.keras.layers.Reshape((1,1,16))(avg_x)
    x = tf.keras.layers.Conv2D(int(channels)//reduction, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='valid')(avg_x)#接着做reduction
    x = tf.keras.layers.Conv2D(int(channels), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='valid')(x)#扩展x回到原来的通道个数
    cbam_feature = tf.keras.layers.Activation('sigmoid', name='cbam')(x)#对x 做 sigmoid 激活
    return feature, cbam_feature #返回以cbam_feature 为scale，对feature做拉伸加权的结果（Excitation）

inputs = tf.keras.Input(shape=(28, 28, 1), name='data')
x = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x)
x, cbam_feature = SeNetBlock(x) # SeNetBlock 对有16个卷积核的第二个卷积层进行加权操作
x = x + cbam_feature
x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(120, activation='relu')(x)
x = tf.keras.layers.Dense(84, activation='relu')(x)
outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='senet_lenet')

#model.summary()
#plot_model(model, to_file='./senet_lenet.png', show_shapes=True,expand_nested=True)

model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs\\", histogram_freq=1)
checkpoint_path = "./checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1, period=1)

model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test), callbacks=[tensorboard_callback, cp_callback])
# tensorboard --logdir=fit_logs

model = tf.keras.models.load_model(checkpoint_path)

cbam_feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('cbam').output, name='cbam_feature') 
cbam_feature_model1 = tf.keras.Model(inputs=inputs, outputs=cbam_feature)
cbam_feature_out = cbam_feature_model.predict(image_d)  # out [?,1,1,16]
cbam_feature_out1 = cbam_feature_model1.predict(image_d)

num = cbam_feature_out.reshape((100, -1)) # 100*16
num1= cbam_feature_out1.reshape((100, -1)) # 100*16
print((num==num1).all())
num = num * 255.
plt.imshow(num, cmap=plt.get_cmap('hot'))  
plt.title('cbam_feature_out')
plt.show()
num1 = num1 * 255.
plt.imshow(num1, cmap=plt.get_cmap('hot'))  
plt.title('cbam_feature_out1')
plt.show()
