# 问题描述
SENetBlock是一个很通用的模块，结构其实非常容易实现，也易于添加到我们自己的网络结构中去，提升现有效果。我们只需要在相关的层后面，添加一个SENetBlock层就可以了。
利用SeNetBlock优化Lenet的训练效果，并观察通道权重的数据特点

# 要求

我们把SeNetBlock放在Lenet第二个卷积层后面（pooling之前），让他对有16个卷积核的第二个卷积层进行加权操作

在mnist上训练好senet_lenet后，如果我们用100个类型连续的0-9样本去测试senet_lenet，会得到cbam_feature（通道权重）如下所示（100x16排列，每一行一个样本，每10行一个类型，每一列是第二层16个卷积特征输出通道对应的权重之一）第二层卷积特征的这16个通道，对所有样本有一定规律，对不同种类型，也有相应的偏好。

+ 如何得到cbam_feature（通道权重）并作图？

让`def SeNetBlock(feature,reduction=4)`返回两个值 `return x，cbam_feature`
这样可以另搞一个 cbam_feature 模型 `cbam_feature_model=model(input,output=cbam_feature )`
然后再组织好数据， 前向一次`cbam_feature_out=cbam_feature_model（data)`，得到cbam_feature_out后，再reshape为100x16的矩阵，然后用plt绘图
