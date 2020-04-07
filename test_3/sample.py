import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D,concatenate,Input,Activation, add
from tensorflow.keras.utils import plot_model
import time
import os

num_classes=10
mnist = tf.keras.datasets.mnist

#1. prepare datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.one_hot(y_train, num_classes)
y_test = tf.one_hot(y_test, num_classes)
x_train=x_train.reshape((60000,-1))
x_test=x_test.reshape((10000,-1))

x_train_1 = x_train[:,:392]
x_train_2 = x_train[:,392:]
x_test_1 = x_test[:,:392]
x_test_2 = x_test[:,392:]

train_ds = tf.data.Dataset.from_tensor_slices(
   ((x_train_1,x_train_2), y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices(((x_test_1,x_test_2), y_test)).batch(64) 

'''
#2. net_build
input_1 = Input(shape=(392,), name='input_1')
input_2 = Input(shape=(392,), name='input_2')
inputs = Input(shape=(392,), name='D1_input')
outputs = Dense(units=num_classes, input_dim=392, name='D1')(inputs)
shared_base = Model(inputs=inputs,outputs=outputs, name='seq1')
s1 = shared_base(input_1)
s2 = shared_base(input_2)
b = K.zeros(shape=(10))
class Bias(keras.layers.Layer):
    def __init__(self, num_outputs , **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.num_outputs = num_outputs
    def build(self, input_shape):
        self.bias = self.add_weight("bias",shape=[self.num_outputs])
        super(Bias, self).build(input_shape)# Be sure to call this somewhere
        #或 self.built = True
    def call(self, input):
        return input + self.bias
s = add([s1, s2])
x = Bias(10,name='bias_add')(s)
o_x = Activation('softmax')(x)
siamese_net = Model(inputs=(input_1, input_2), outputs=o_x)
plot_model(siamese_net, to_file='./siamese_net.png', show_shapes=True,expand_nested=True)
print([var.name for var in siamese_net.trainable_variables])
#3. train and test
loss_object = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.SGD(0.01)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
checkpoint_path = "./checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1, period=1)
siamese_net.summary()
siamese_net.compile(optimizer=optimizer,loss=loss_object, metrics=[train_accuracy])
#siamese_net.fit([x_train_1,x_train_2], y_train, epochs=5, callbacks = [cp_callback])
siamese_net.fit(train_ds, epochs=5, callbacks = [cp_callback])
loss,acc= siamese_net.evaluate(train_ds)
print("saved model, loss: {:5.2f}, acc: {:5.2f}".format(loss,acc))
'''

#try SparseCategoricalCrossentropy without one-hot
loss_object = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.SGD(0.01)
train_loss = tf.keras.metrics.Mean(name='train_loss')
#try SparseCategoricalAccuracy
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

#try not use tf.function to debug
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = siamese_net(images)
        #print(predictions)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, siamese_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, siamese_net.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = siamese_net(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

#4. draw weights of 10 classes
train_weights=siamese_net.get_layer('seq1').get_layer('D1').kernel.numpy()


num = np.arange(0, 392, 1, dtype="float")
num = num.reshape((14, 28))
plt.figure(num='Weights', figsize=(10, 10))  # 创建一个名为Weights的窗口,并设置大小
for i in range(10):  # W.shape[1]
    num = train_weights[:, i: i+1].reshape((14, -1))
    plt.subplot(2, 5, i + 1)
    num = num * 255.
    plt.imshow(num, cmap=plt.get_cmap('hot'))  
    plt.title('weight %d image.' % (i + 1))  # 第i + 1幅图片
plt.show()
print(np.min(num))
print(np.max(num))
