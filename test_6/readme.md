在Lenet中，分别使用ReLU及sigmoid激活函数，观察不同情况下，Lenet学习MNIST分类时，参数的变化。

在最终训练好Lenet的情况下，观察分类操作前的最后一个全连接层fc2的84位特征输出向量，比较不同类型样本的fc2特征图。

（1）tensorboard可视化包括：loss， acc， w、b参数的分布，至少比较2种情况：[ReLU, sigmoid] 。
有兴趣可以尝试分别使用GradientDescentOptimizer 和AdamOptimizer的效果。

（2）fc2特征图不用tensorboard显示，plot绘出即可：
在模型训练好的情况下：绘制10个数字类型的fc2特征图，
每个类型一张fc2图，共10张fc2图：
每一张fc2图由同类型的100个不同样本的fc2特征按行拼接组成。
例如数字3的fc2特征图，由100个不同的数字3样本的fc2特征按行拼接组成。
故一张fc2图的大小为：100（行）*84（列）。

2.
特征分布可视化
使用tensorboard中的embedding projector工具，对lenet模型提取的数据特征进行可视化
要求：
对比mnist原始数据embedding和训练好的lenet的f2特征embedding效果，感受特征学习的效果。
参考：https://github.com/efeiefei/tensorflow_documents_zh/blob/master/get_started/embedding_viz.md
