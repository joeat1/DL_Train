## 问题描述
+ **采用迁移学习方法，用预训练的mobilenetv2模型进行分类、关键点预测多任务学习**
## 要求
+ 通过0—360度旋转构建飞机数据集, 每个飞机360个样本，旋转1度一个样本
+ 其中，80%的数据用来训练，10%用来做验证集，10%用来做测试集。思考通过优化网络结构及loss函数，提示网络训练结果。
+ 不调式预训练模型参数，通过构建合适的网络结构及选择合适的loss函数，提升模型的分类、及关键点标注效果。尽量优化模型效果，作业成绩根据模型效果评判。
## 提示
+ 选择较小学习率，迭代可能多一些，当然如果设计有自己特色的网络，根据需求，可调整。
+ 过拟合：不合适的结构可能导致明显的过拟合，注意训练精度和验证精度的差别（实际在训练过程中还会出现大量波动），以及lm 精度之间的巨大差距。

## 定性分析

+ BatchNormal 层 实现 零均值化，归一化，去相关 对过拟合的效果有益处，添加 BatchNormal 层与不添加相比，训练结果更为平稳，过拟合影响较轻；
+ GlobalAveragePooling2D 层对图像分类工作的效果较好
+ 在迁移学习中，自定义附加的全连接层，建议选定激活函数，否则训练效果一般

## 降低过拟合——正则化
+ 调整目标函数 使得难例得到重视
+ 参数正则化   网络在降低样本损失的同时，选择较小的参数权值
+ dropout层   消除节点相互依赖
