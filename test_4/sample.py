
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D,concatenate,Input,Activation, Lambda
from tensorflow.keras.utils import plot_model
from sklearn import metrics
import time
import os

num_classes=10
mnist = tf.keras.datasets.mnist

#查看mnist 图片
def minist_draw(im):
    im = im.reshape(28, 28)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.axis('off')
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.savefig("test.png")
    plt.close()

#batch_x MNIST样本 batch_y, MNIST标签 num_cls （数字类型个数，10，为了让10个数字类型都充分采样正负样本对）
def balanced_batch(batch_x, batch_y, num_cls=10):
    batch_size=len(batch_y)
    pos_per_cls_e = round(batch_size/2/num_cls/2)
    pos_per_cls_e*=2
    index=batch_y.argsort()
    ys_1 = batch_y[index] # 按照0-9顺序排序好，index为原始下标
    #print(ys_1)
    num_class=[]     # 各类别的数量
    pos_samples=[]   # 包含正例子的下标号，临近的两个作为一组
    neg_samples=set()# 
    cur_ind=0
    for item in set(ys_1):
        num_class.append((ys_1==item).sum())
        num_pos=pos_per_cls_e
        while(num_pos>num_class[-1]):
            num_pos-=2
        pos_samples.extend(np.random.choice(index[cur_ind:cur_ind+num_class[-1]],num_pos,replace=False).tolist()) #每个类选num_pos个数字组成num_pos/2对正例
        neg_samples=neg_samples|(set(index[cur_ind:cur_ind+num_class[-1]])-set(list(pos_samples))) #去掉大量同类后，余下是少量各类，遍历列表 容易 得到负例
        cur_ind+=num_class[-1] #游标移到下一个类别的数据上
    neg_samples=list(neg_samples)
    x1_index=pos_samples[::2]
    x2_index=pos_samples[1:len(pos_samples)+1:2]
    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples)+1:2])
    p_index=np.random.permutation(len(x1_index)) #打乱数据集
    x1_index=np.array(x1_index)[p_index]
    x2_index=np.array(x2_index)[p_index]
    r_x1_batch=batch_x[x1_index]
    r_x2_batch=batch_x[x2_index]
    r_y_batch=np.array(batch_y[x1_index]!=batch_y[x2_index],dtype=np.float32)
    #return r_x1_batch, r_x2_batch,r_y_batch 输入样本对（ r_x1_batch, r_x2_batch）样本对正负例标签 r_y_batch
    return r_x1_batch,r_x2_batch, r_y_batch #三个numpy类型

def batch_iter(x, y, batch_size=64):
    np.random.shuffle(x)
    np.random.shuffle(y) #打乱数据集(X，Y)
    num_examples = len(x)
    num_batch = num_examples // batch_size
    x_t1 = []
    x_t2 = []
    y_t = []
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, num_examples)
        r_x1_batch,r_x2_batch, r_y_batch = balanced_batch(x[start_id:end_id], y[start_id:end_id])
        x_t1.append(r_x1_batch)
        x_t2.append(r_x2_batch)
        y_t.append(r_y_batch)
    x_t1 = np.vstack(x_t1)
    x_t2 = np.vstack(x_t2)
    y_t = np.hstack(y_t)
    return (x_t1, x_t2), y_t

def pre_data(images, labels, size, print_t=False):
    # 构建平衡测试集
    len_data = len(images)
    from_0_to_9 = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len_data): #对数据集进行遍历
        for j in range(10): #对0-9进行遍历
            if labels[i]== j:
                from_0_to_9[j].append(images[i])
    image1 = []  # 存在第一个图像
    image2 = []  # 存放第二个图像
    label = []  # 存在label
    count_list_right = {}  # 设计为字典
    count_list_nega = {}  # 设计为字典
    #下面考虑正例，每个数字占（size/2）的1/10
    num_each_right = int(size/2/10)  #每个数字创造的样本数
    num_each_nega = int(size / 2 / 45)
    for i in range(10):  #对每个数字做遍历
        isbreak = 0
        count =0 #当前样本数为0
        len_i = len(from_0_to_9[i]) # 获取当前这个数字对应的样本数量
        for j in  range(len_i): # 对这个数字对应的样本做遍历
            if isbreak:
                break
            for k in range(len_i): # 对这个数字对应的样本进行二重遍历
                if j==k :
                    continue
                image1.append(from_0_to_9[i][j])
                image2.append(from_0_to_9[i][k])
                label.append([0])
                count+=1
                if count>=num_each_right:
                    count_list_right["(%d,%d)"%(i,i)]=count
                    isbreak = 1
                    break    #跳出两层for循环
    #下面考虑负样本
    for i in range(10): # 对0-9遍历
        for j in range(i+1,10): #仍对0-9遍历
            for count in range(num_each_nega): #构造这num_each_nega多个样本数据
                image1.append(from_0_to_9[i][count])
                image2.append(from_0_to_9[j][count])
                label.append([1])
            count_list_nega["(%d,%d)"%(i,j)]=num_each_nega
    if print_t == True:
        print("正例：",count_list_right)
        print("正例的总样本数为",len(count_list_right)*num_each_right)
        print("反例：",count_list_nega)
        print("正例的总样本数为", len(count_list_nega) * num_each_nega)
        print("测试集长度为：", len(test_data[0]))
    image1 = np.array(image1)
    image2 = np.array(image2)
    label = np.array(label, dtype=np.float32)
    return image1, image2, label

#1. prepare datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train=x_train.reshape((60000,-1))
x_test=x_test.reshape((10000,-1))

train_image1, train_image2, train_label = pre_data(x_train, y_train, size=9000)
test_image1, test_image2, test_label = pre_data(x_test, y_test, size=9000)

#2. net_build
input_1 = Input(shape=(784,), name='input_1')
input_2 = Input(shape=(784,), name='input_2')

dense = Dense(units=500, input_dim=784, activation='sigmoid', name='dense')
dense1 = Dense(units=10, input_dim=500, activation='relu', name='dense1')#, activation='relu', name='dense1')

hidden_a = dense1(dense(input_1))
hidden_b = dense1(dense(input_2))

distence = K.sqrt(K.sum(tf.square(hidden_a - hidden_b), axis=1))
model = Model(inputs=(input_1, input_2), outputs=distence)

#plot_model(model, to_file='./siamese_net2.png', show_shapes=True,expand_nested=True)

#3. train and test
Q = 6
def custom_loss(label_pair, e_w):
    loss_p = (1 - label_pair) * 2 * e_w ** 2 / Q 
    loss_n = label_pair * 2 * Q * K.exp(-2.77 * e_w / Q)
    loss = K.mean(loss_p + loss_n)
    return loss

auc_ = tf.keras.metrics.AUC()
def auc(label_pair, e_w):
    e_w = tf.keras.layers.Flatten()(e_w)
    e_w = (e_w - K.min(e_w))/(K.max(e_w) - K.min(e_w))
    label_pair = tf.keras.layers.Flatten()(label_pair)
    return auc_(label_pair, e_w)

checkpoint_path = "./checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1, period=1)

#model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.03),loss=custom_loss, metrics=[auc])

model.fit((train_image1, train_image2), train_label, epochs=10)#, callbacks = [cp_callback])

#4
loss, auc= model.evaluate((test_image1, test_image2), test_label)
print("saved model, loss: {:5.2f}, acc: {:5.2f}".format(loss, auc))

e_w = model.predict((test_image1, test_image2))
#print(type(e_w)) #<class 'tensorflow.python.framework.ops.EagerTensor'>
threshs = [0.2, 0.5, 0.7 ,1., 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,5.0]
area_list = []
for thresh in threshs:
    predictions = (e_w > thresh) #e_w=e_w/max(e_w)
    precision, recall, th = metrics.precision_recall_curve(test_label, predictions)
    area_list.append(metrics.auc(recall, precision))
    plt.plot(recall, precision, linewidth=1.0,label='thresh='+str(thresh))
plt.plot([0.5,1], [0.5,1], linewidth=1.0,label='equal')
plt.title("precision and recall curve")
plt.legend()
plt.xlabel("recall")
plt.ylabel('precision')
plt.show()
print(threshs, area_list)
