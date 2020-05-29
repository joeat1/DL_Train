# coding=utf-8
import tensorflow as tf
import os
import json
import numpy as np
#用下面的环境配置指定不用gpu，只用cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# **************************** 创建一个分发变量和图形的策略 ***********************************

#TODO 1:
#定义双节点的分布式训练策略及相应环境变量
#最好就像这样，在程序最开始就定义并行分布训练策略，否则一些环境冲突下，策略实例化的启动会报错

os.environ["TF_CONFIG"] = json.dumps({
    "cluster":{
      "worker" : ["localhost:8001", "localhost:8002"],
    },
    "task":{"type": "worker", "index": 1}
  })
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


#******************************准备数据**********************************************
mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 向数组添加维度 -> 新的维度 == (28, 28, 1)
# 我们这样做是因为我们模型中的第一层是卷积层
# 而且它需要一个四维的输入 (批大小, 高, 宽, 通道).
# 批大小维度稍后将添加。
train_images = train_images[..., None]
test_images = test_images[..., None]

# 获取[0,1]范围内的图像。
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

#***************************** 设置输入流水线 **********************************
BUFFER_SIZE = len(train_images)
num_workers = 2
# 设置 batch size (全局bs = 单卡bs * num_gpus)
# 单卡bs
BATCH_SIZE_PER_REPLICA = 64 
# 全局bs
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_workers

# 创建数据集并分发它们：
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 

num_epochs = 5
learning_rate = 0.001

def model_build():
	model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
	return model

	
#TODO 2:完成strategy 环境下的相应工作

with strategy.scope():
  model = model_build()
  model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
steps_per_epoch=np.ceil(60000/GLOBAL_BATCH_SIZE)
validation_steps=np.ceil(10000/GLOBAL_BATCH_SIZE)
model.fit(train_dataset.repeat(), epochs=num_epochs,steps_per_epoch = steps_per_epoch,validation_data=test_dataset.repeat(),
                    validation_steps=validation_steps)