假设有函数y = cos(ax + b), 其中a为学号倒数第5和第4位，b为学号最后两位。首先从此函数中以相同步长（点与点之间在x轴上距离相同），在0<(ax+b)<2pi范围内，采样出2000个点，然后利用采样的2000个点作为特征点进行三次函数拟合(三次函数形式为 y = w1 * x + w2 * x^2 + w3 * x^3 + b, 其中wi为可训练的权值，b为可训练的偏置值，x和y为输入的训练数据 ) 。要求使用TF2.x,用model.fit和自定义循环两种训练方法实现三次函数拟合的全部流程。拟合完成后，分别使用回调函数和model.save模式保存拟合的模型。然后，针对两种模型存储方式分别编写模型恢复程序分别，并同时绘制图像（图像包括所有的2000个采样点及拟合的函数曲线）。保存和恢复的都是最后一次训练的模型，记录和打印保存前和恢复后的loss,并查看是否一致。

三个python脚本文件：1. modelfit函数拟合及模型保存程序modelfit_todo.py2. 自定义循环拟合程序custom_todo.py，及其对应的 model.save模型恢复程序loadmodel_todo.py（loadmodel.py直接恢复模型结构及参数，不要再重复定义模型）。


提示：
test data就是train data
学习率推荐大一些，例如0.1；迭代epoch可以多一些，例如最多2000左右
尝试并报告loss_object = tf.keras.losses.MeanSquaredError()改为 tf.keras.losses.MeanSquaredError（去括号）的效果，为什么？
尝试并报告自定义循环添加@tf.function和不添加的效果，可调试否？运行时间？
