import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import Sequential,Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D
import time
import os
#if you have omp problem in mac
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# data prepare
#
a = 20
b = 20
x_t = np.arange(-b/a , (2*np.pi-b)/a , 2*np.pi/2000/a,dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis = 1)
y_train = np.cos(a * x_t +  b)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# define model
inputs = tf.keras.Input(shape=(3,), name='data')
outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Lm2')


#try "tf.keras.losses.MeanSquaredError" without "()" as well?
loss_object = tf.keras.losses.MeanSquaredError()

#optimizer = tf.keras.optimizers.Adam(0.37)
optimizer = tf.keras.optimizers.SGD(0.3)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

#try not use tf.function to debug,time?
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_object(labels, predictions) #tf.reduce_mean(tf.square(predictions - labels))#
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)

@tf.function
def test_step(data, labels):
    predictions= model(data)
    t_loss = loss_object(labels, predictions)#tf.reduce_mean(tf.square(predictions - labels))#

    test_loss(t_loss)

EPOCHS = 2000

for epoch in range(EPOCHS):
    start = time.time()
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    test_loss.reset_states()
    for data, labels in train_dataset.shuffle(buffer_size = 10000).batch(32) :  #batchsize = 64，一批64，使用batch的时候一定需要先进行数据的打乱，否则会出现局部最优，而使得训练效果越来越差
       train_step(data, labels)
    for data, labels in train_dataset.shuffle(buffer_size = 10000).batch(32):
       test_step(data, labels)
    #train_step(x_train, y_train) # 全部放进去
    #test_step(x_train, y_train)
    end = time.time()
    template = 'Epoch {}, Loss: {:.3f},  Test Loss: {:.3f}，Time used: {:.2f}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         test_loss.result(),end-start))
    if epoch%200==199:
        #model save here
        model.save_weights('./checkpoints/my_checkpoint')
        print(model.variables)

model.compile(loss='mse')
loss= model.evaluate(x_train, y_train)
print("Saved model, loss: {:5.2f}".format(loss))
#model save here
model.save("./checkpoints/saved_model.h5")

predictions=model(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original values')
plot2 = plt.plot(x_t, predictions, 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.show()
