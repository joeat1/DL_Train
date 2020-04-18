import os
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
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorboard.plugins import projector #tf2.x
#from tensorflow.contrib.tensorboard.plugins import projector #tf1.x

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
    label_d = []
    cur_ind=0
    for i in range(10):
        image_d.extend(images[index[cur_ind:cur_ind+num_each]])
        label_d.extend([i]*num_each)
        cur_ind += num_each_all[i]
    image_d = np.array(image_d)
    return image_d, label_d
imgs_embemdding, labels_embemdding = pre_data(x_test, y_test, size=1000)

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

activation_type = 'relu' # 'sigmoid' #
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

feature_model = tf.keras.Model(inputs=inputs, outputs=model.get_layer('fc2').output, name='fc2_feature')
feature_embemdding = feature_model.predict(imgs_embemdding)  # out [?,84] imgs_embemdding [28,28,1]

#print(feature_embemdding.shape)

# 对比mnist原始数据embedding和训练好的lenet的f2特征embedding效果，感受特征学习的效果。
# 特征（所作的mnist原始数据对比是用mnist 784 图像向量直接作特征）
LOG_DIR = "projector_demo"  # Tensorboard log dir # tensorboard --logdir=projector_demo  # http://localhost:6006/#projector
METADATA_FNAME = "meta.tsv"  # Labels will be stored here
SPRITES_FNAME = "mnistdigits.png"

def register_embedding(feature, img_data, labels, log_dir) -> None:
    #在register_embedding注册中要作2个事情：1. 保存特征变量在ckpt文件，2. 配置projector_config.pbtxt 文件。
    #给配置文件准备个文件夹
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #准备一个变量去保存embedding数据三元组里的特征
    #名字必须是embedding,不要问我为什么，也不要问我怎么知道的
    embeding_name="embedding"
    embedding_var=K.variable(feature,name=embeding_name)
    # 保存embedding变量在 “1. ckpt文件”
    # 用checkpoint保存embedding配置信息，指定其embedding是我们创建的embedding变量
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(log_dir, "em.ckpt"))
    #创建 sprite 图像文件，label的meta文件
    sprite_and_meta_writer(img_data,labels,log_dir)
    # 配置投影 “2.projector_config.pbtxt”
    config = projector.ProjectorConfig()
    # 由于embedding是复合类型，这儿需要调用add()方法实例化一个embedding
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    #一定要加`/.ATTRIBUTES/VARIABLE_VALUE`！！！！！
    embedding.tensor_name = embeding_name+"/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = METADATA_FNAME
    embedding.sprite.image_path = SPRITES_FNAME #'mnistdigits.png'
    embedding.sprite.single_image_dim.extend([img_data.shape[1],img_data.shape[2]])
    projector.visualize_embeddings(log_dir, config)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits

def sprite_and_meta_writer(img_data,labels,log_dir):
    # 绘制sprite和meta文件 
    # 在注册中调用sprite和meta的创建的好处是，可以在不同的迭代，用不同的数据，创建不同的log_dir保存不同训练阶段的特征投影效果。 
    # 若只用一批固定数据作对比，可以将sprite和meta文件从register_embedding单独提出，在前期只做一次即可（但注意要修改一下writer，要给不同比较特征不同的log_dir路径）
    to_visualise = img_data
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)
    sprite_image = create_sprite_image(to_visualise)
    plt.imsave(os.path.join(log_dir, SPRITES_FNAME),sprite_image,cmap='gray')
    plt.imshow(sprite_image,cmap='gray')
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, METADATA_FNAME), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

register_embedding(feature_embemdding, imgs_embemdding, labels_embemdding, LOG_DIR)
