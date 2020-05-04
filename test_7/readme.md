 grad-cam

使用Guided Grad-CAM算法对预训练分类网络VGG16进行可视化评估，深入理解梯度、反向传播、网络特征等物理含义，了解TF2中采用自定义方法修改梯度的过程。

要求：对预训练分类网络VGG16进行效果可视化，选一张图片进行某类型的效果展示。提交源码及可视化结果图片。多分类结果，比较top5中前几种有区别的类型，看他们的grad cam和guided grad cam都在图像中指向了哪里？



Guided Grad-CAM将预训练模型中，某个特征层f 对输入图像img分类为c的图像依据以“热点图”展示出来。例如我们输入一张猫的图像，分类模型预测为“猫”,通过可视化我们可以发现是图像哪些部分支持这个分类结果，尖耳朵？胡须？还是猫脸？这些区域对分类的支持程度怎样？支持越高的区域，其在“热点图”上就越“热”。

Guided Grad-CAM分为两部分：1. Guided-backpropagation 梯度图 与 2. Grad-CAM(Class Activation Mapping) 特征图。他们分别从不同的角度描述特征层f对图像分类的依据。其中，Guided-backpropagation 梯度图 从梯度的意义进行解释（输出对输入图像的梯度）：最能引起结果变化的图像区域分布（并且这些区域该“+”还是该“-“）；Grad-CAM 特征图 从feature map对结果的贡献进行解释：对该次分类结果，决策过程对不同feature的采纳程度，这个程度我们视为该类型对该feature map的“权重”。Grad-CAM即是加权的feature map。而Guided Grad-CAM 是 Guided-backpropagation 梯度图 与 Grad-CAM最终融合（相乘）的结果，显示从这两个角度都重要的图像分类依据。

与一般反向传播求梯度的过程唯一不同的地方在于对激活节点ReLU的传播处理, Guided-backpropagation梯度只传播“正”的梯度。

