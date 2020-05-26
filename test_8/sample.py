# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
import os
import os.path as osp
import math
import random
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 160 # All images will be resized to 160x160
ORIGINAL_IMG_SIZE=255
scale_f=IMG_SIZE/ORIGINAL_IMG_SIZE
LM_POINTS=['Nose','Fuselage','Empennage','FLwing','FRwing','BLwing','BRwing']

data_root = pathlib.Path(r'F:\study\new_class\deeplearning\test1-fly\AIRPLANES')
print(data_root)

def mk_ap_dataset(data_root):

    #列出可用的类别标签：
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_names
    
    #为每个类别标签分配索引：
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    index_to_label = dict((index, name) for index, name in enumerate(label_names))

    all_image_paths = list(data_root.glob('*/*jpg'))
    all_anno_paths = list(data_root.glob('*/*txt'))
    anno_to_img= dict( (p_img,p_anno) for p_img in all_image_paths for p_anno in all_anno_paths if p_img.stem==p_anno.stem)
 
    c_lables=np.zeros(len(anno_to_img))
    lm_labels=np.zeros((len(anno_to_img),14))
    index=0
    for img,anno in anno_to_img.items():
        with open(anno) as f_anno:
            lines = f_anno.readlines()
            c_lables[index]=label_to_index[img.parent.name]
            id_line=0
            for line in lines:
                name_id,x,y = line.split()
                #推荐：将关键点标签分布范围从原数据的[0~ORIGINAL_IMG_SIZE]变为：[-IMG_SIZE/2~IMG_SIZE/2]
                #若如此，后面显示图像验证绘图的时候，要注意把关键点分布转为图像显示需要的[0~IMG_SIZE ]
                lm_labels[index][id_line*2]=(int(x)/ORIGINAL_IMG_SIZE*2-1)*IMG_SIZE/2
                lm_labels[index][id_line*2+1]=(int(y)/ORIGINAL_IMG_SIZE*2-1)*IMG_SIZE/2
                id_line+=1
        index+=1
    all_image_paths = [str(path) for path in all_image_paths]
    return all_image_paths,c_lables,lm_labels,index_to_label

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image/128.0-1  # normalize to [-1,1] range

    return image

def deprocess_image( img ):
    img = (img + 1.0)*128
    return img

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


all_image_paths,all_cls_labels,all_lm_labels,index_to_label=mk_ap_dataset(data_root)


#可以在创建数据集之前就先打乱原数据
state = np.random.get_state()
np.random.shuffle(all_image_paths)
np.random.set_state(state)
np.random.shuffle(all_cls_labels)
np.random.set_state(state)
np.random.shuffle(all_lm_labels)
len_data=len(all_image_paths)

def map_samples(path, cls_labels,lm_labels):
    return load_and_preprocess_image(path), {'c_pred':cls_labels,'lm_pred':lm_labels}

train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[0:int(len_data*0.8)],
                                         all_cls_labels[0:int(len_data*0.8)],
                                         all_lm_labels[0:int(len_data*0.8)])).map(map_samples)
val_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[int(len_data*0.8):int(len_data*0.9)],
                                         all_cls_labels[int(len_data*0.8):int(len_data*0.9)],
                                         all_lm_labels[int(len_data*0.8):int(len_data*0.9)])).map(map_samples)
test_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[int(len_data*0.9):],
                                         all_cls_labels[int(len_data*0.9):],
                                         all_lm_labels[int(len_data*0.9):])).map(map_samples)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train_ds.repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_batches = val_ds.repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

train_batches = train_batches.prefetch(buffer_size=AUTOTUNE)
val_batches = val_batches.prefetch(buffer_size=AUTOTUNE)
test_batches = test_batches.prefetch(buffer_size=AUTOTUNE)

#做好数据集后，看看数据读取有没有什么问题，看看类型，关键点标签是否有错误，image是不是有问题
'''
for image_batch, label_dict in test_batches.take(1):
    imgs=image_batch.numpy()

    for i in range(3):
        c_label=label_dict['c_pred'].numpy()
        lm_label=(label_dict['lm_pred'].numpy()+IMG_SIZE/2).astype(int)
        plt.figure()
        
        ax=plt.subplot(1,1,1) 
        plt.title(index_to_label[int(c_label[i])])
        for j in range(len(LM_POINTS)):
            ax.scatter(lm_label[i][j*2],lm_label[i][j*2+1])
            ax.text(lm_label[i][j*2]*1.01, lm_label[i][j*2+1]*1.01, LM_POINTS[j], fontsize=10, 
                color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签
        plt.imshow(np.uint8(deprocess_image(imgs[i])))
        plt.show()
'''

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
mobile_net = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False,weights=None)
mobile_net.load_weights('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5')
mobile_net.summary()
#注意：我们不调试预训练模型参数
mobile_net.trainable=False

#TODO 1:优化迁移多任务学习网络结构
#一个简单的多任务迁移学习model构建例子，可以尝试分类、关键点标注效果，
#但是效果不一定很好，考虑修改自己的模型结构，提升分类和关键点标注效果
x=mobile_net.outputs[0]
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(120, activation='relu', name='features')(x)

#另起一支：注意，用get_layer取中间层的适合，是.output,后面没有[0]
y=mobile_net.get_layer('Conv_1_bn').output #假设关键点标注效果是成功识别图片类别的一个重要原因需要寻找得到关键识别出各模式点的层后进行再次训练
y=tf.keras.layers.GlobalAveragePooling2D()(y) #展开成一维数据
y=tf.keras.layers.Dense(320, name='key_nodes')(y)
y=tf.keras.layers.Dense(120, name='key_nodes_1')(y)

c_pred=tf.keras.layers.Dense(3, activation = 'softmax', name='c_pred')(x)
lm_pred=tf.keras.layers.Dense(14, name='lm_pred')(y)
mt_model=tf.keras.Model(inputs=mobile_net.inputs, outputs=[c_pred,lm_pred], name='mt_model')
mt_model.summary()

#TODO 2:完成多任务的model compile
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True)
mt_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss={
                  'c_pred': 'sparse_categorical_crossentropy',
                  'lm_pred': 'mae'}, #mse （平方误差）和mae（绝对值误差）
              loss_weights={
                  'c_pred': 1.,
                  'lm_pred': 1.
              }, metrics={'c_pred': 'accuracy', 'lm_pred': 'accuracy'}) 
steps_per_epoch=np.ceil(len_data*0.8/BATCH_SIZE)
v_steps_per_epoch=np.ceil(len_data*0.1/BATCH_SIZE)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="flight_mt_logs\\", histogram_freq=1)


mt_model.fit(train_batches,epochs=50, verbose=2, validation_data=val_batches,
             steps_per_epoch=steps_per_epoch,validation_steps=v_steps_per_epoch, callbacks=[tensorboard_callback])


#在测试集上评测模型效果
result = mt_model.evaluate(test_batches)
result_dict=dict(zip(mt_model.metrics_names, result))


#挑几张训练集上的数据，对比标签和模型分类及关键点标注效果
#后面会看测试集上的数据对比，如果有过拟合发生，训练和测试集上的对比差距会很大
for image_batch, label_dict in train_batches.take(1):
    imgs=image_batch.numpy()
    output=mt_model(image_batch)
    c_pred=output[0].numpy()
    lm_pred=(output[1].numpy()+IMG_SIZE/2)
    for i in range(10):
        c_label=label_dict['c_pred'].numpy()
        lm_label=(label_dict['lm_pred'].numpy()+IMG_SIZE/2)
        plt.figure()
        
        ax1=plt.subplot(1,2,1) 
        plt.title(index_to_label[int(c_label[i])])
        for j in range(len(LM_POINTS)):
            ax1.scatter(lm_label[i][j*2],lm_label[i][j*2+1])
            ax1.text(lm_label[i][j*2]*1.01, lm_label[i][j*2+1]*1.01, LM_POINTS[j], fontsize=10, 
                color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签
        plt.imshow(np.uint8(deprocess_image(imgs[i])))
        
        
        ax2=plt.subplot(1,2,2)
        plt.title(index_to_label[np.argmax(c_pred[i])])
        for j in range(len(LM_POINTS)):
            ax2.scatter(lm_pred[i][j*2],lm_pred[i][j*2+1])
            ax2.text(lm_pred[i][j*2]*1.01, lm_pred[i][j*2+1]*1.01, LM_POINTS[j], fontsize=10, 
                color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签  
        plt.imshow(np.uint8(deprocess_image(imgs[i])))
        plt.show()

#再挑几张测试集上的数据，对比标签和模型分类及关键点标注效果
for image_batch, label_dict in test_batches.take(1):
    imgs=image_batch.numpy()
    output=mt_model(image_batch)
    c_pred=output[0].numpy()
    lm_pred=(output[1].numpy()+IMG_SIZE/2)
    for i in range(10):
        c_label=label_dict['c_pred'].numpy()
        lm_label=(label_dict['lm_pred'].numpy()+IMG_SIZE/2)
        plt.figure()
        
        ax1=plt.subplot(1,2,1) 
        plt.title(index_to_label[int(c_label[i])])
        for j in range(len(LM_POINTS)):
            ax1.scatter(lm_label[i][j*2],lm_label[i][j*2+1])
            ax1.text(lm_label[i][j*2]*1.01, lm_label[i][j*2+1]*1.01, LM_POINTS[j], fontsize=10, 
                color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签
        plt.imshow(np.uint8(deprocess_image(imgs[i])))
        
        
        ax2=plt.subplot(1,2,2)
        plt.title(index_to_label[np.argmax(c_pred[i])])
        for j in range(len(LM_POINTS)):
            ax2.scatter(lm_pred[i][j*2],lm_pred[i][j*2+1])
            ax2.text(lm_pred[i][j*2]*1.01, lm_pred[i][j*2+1]*1.01, LM_POINTS[j], fontsize=10, 
                color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签  
        plt.imshow(np.uint8(deprocess_image(imgs[i])))
        plt.show()
