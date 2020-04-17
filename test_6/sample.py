import time
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
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

n_classes = 10

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
        image_d.extend(images[index[cur_ind:cur_ind+num_each]])
        cur_ind += num_each_all[i]
    image_d = np.array(image_d)
    return image_d
image_d = pre_data(x_test, y_test, size=1000)

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

activation_type = 'sigmoid' #'relu' # 
model_name = "lenet−{}-{}".format(activation_type, int(time.time()))


inputs = tf.keras.Input(shape=(28, 28, 1), name='data')
x = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x)
x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(120, activation=activation_type)(x)
#p1. all zeros initialization
#x = tf.keras.layers.Dense(84, activation='relu',kernel_initializer='zeros',bias_initializer='zeros')(x)
x = tf.keras.layers.Dense(84, activation=activation_type, name='fc2')(x) # 全连接层fc2

#p2. no softmax
#outputs = tf.keras.layers.Dense(n_classes)(x)
outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lenet')
plot_model(model,'lenet.png',show_shapes=True,expand_nested=True)

#model.summary()

model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), #GradientDescentOptimizer：tf.keras.optimizers.SGD
        #optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), #AdamOptimizer：tf.keras.optimizers.Adam
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="test_logs\\{}".format(model_name), histogram_freq=1)

history =model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

fc2_feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fc2').output, name='fc2_feature')
fc2_feature_out = fc2_feature_model.predict(image_d)  # out [?,84]

#print(fc2_feature_out.shape)

plt.figure(num='fc', figsize=(10, 10))  # 创建一个名为Weights的窗口,并设置大小
for i in range(10): 
    num = fc2_feature_out[i*100:i*100+100, :]
    plt.subplot(2, 5, i + 1)
    num = num * 255.
    plt.imshow(num, cmap=plt.get_cmap('hot'))  
    plt.title(i)  # 数值i
plt.savefig("{}-fc2.png".format(model_name))
plt.show()
