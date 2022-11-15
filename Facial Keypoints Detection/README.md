# 1. 面部关键点检测比赛

地址：[Facial Keypoints Detection](https://www.kaggle.com/competitions/facial-keypoints-detection/leaderboard)

参考复现地址：
1. [Facial Keypoint Detection with fastai v2](https://www.kaggle.com/code/jacobfrantz/facial-keypoint-detection-with-fastai-v2)
2. [Data Augmentation for Facial Keypoint Detection](https://www.kaggle.com/code/balraj98/data-augmentation-for-facial-keypoint-detection)


# 2. 方案

这是一个典型的CV的比赛，需要在图象中检测出面部的关键点。

input: 96\*96\*1的图像。

Target: 15\*2的坐标点

由于Target维度是固定的，所以这是一个回归拟合的问题。

学习的这两套方案的流程：

| 步骤 |目标|内容|
|:--:|:--:|:--|
|数据预处理|检查数据|1. 检查图象是否存在空值、缺失值，存在则进行缺失值填充。<br> 2. 检测数据类型：一般使用PIL库进行图象转换，该库只支持两种格式，0-255的uint8格式数据，和0-1的float格式数据。不是这两种格式的数据需进行转换。且需保证训练、测试集保持一致。|
|数据增强|对数据进行基础变换，增加样本量和多样性|1. 图片翻转<br>2. 图片旋转<br>3. 图片明暗度<br>4. 图片水平位移<br>5. 图片随即增加噪声|
|模型|常用的卷积模型(ResNet, Yolo等)、Transformer模型||


# 3. 总结一下

这个比赛比较简单，所以针对模型的使用并不复杂，更多的是在对数据进行增强。同时图像也需要考虑预处理，比如空值、缺失值等等。

模型方案，可以采用简单的FastAI模型，可以快速搭建方案并进行实验。


但是存在疑问：
1. 数据增强是否是有益的？比如增加噪声这个。如果不是一个对抗比赛的话，感觉增加噪声会让效果更差。
2. 这个填充方式似乎有单粗暴，似乎可以用个模型简单预测下，效果是否会提升。
3. 在使用预训练模型的时候，不需要指定预训练的数据集吗？我使用FastAi时，没有指定预训练的权重，好像效果也还行。专门在人脸上做的预训练效果是否更好。



# 4. 补充


## 4.1. 缺失值处理方案


### 4.1.1. 找到缺失值：
1. 区分缺失值：存在形式：None，空，特殊值（-1，-999）
4.2. 代码：
---
1. 找到缺失值

```python
df.isnull().sum()
df.isna()
df[df.isna().T.any()] # 展示存在缺失值的列
```
2. 画图展示所有缺失列:
```python
missing = Train_data.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
sns.histplot(missing)
```
---


### 4.2.1. 缺失值解决思路：

1. 当缺失值数量很小的时候，可以选择填充。如果使用树模型，也可以直接空缺。（树模型对缺失值不敏感）
2. 如果缺失值过多，可以直接删除

对缺失的值进行填充：

1. 类别特征：进行填充众数、填充新类
2. 数值特征：填充平均数、中位数、众数、最大值、最小值
   1. 代码: ```df.fillna(values)```
   2. 注意替换mode时:```df.apply(lambda col:col.fillna(col.mode()[0]))```
3. 有序数据：next、previous。使用前（后）一个有效数字进行填充
   1. 代码```df.fillna(method = 'ffill')```
4. 模型预测填充：对含有缺失值的那一列进行建模并预测。
   1. 如[KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
5. 根据字段含义进行填充。比如大部分人不信教，宗教可以填充为不信教。


## 4.3. FastAI简略教程

简略教程: [FastAI教程](https://qxmsb3wqkm.feishu.cn/docx/Ssn2doqAZoaMpfxBQxHc7ZWLn9b)

简单总结一下：
1. fastai比较方便，但是定制化稍显麻烦。
2. fastai文档很烂，查起来有难度。


## 4.4. PIL库
PIL 是第三方库python数据处理库。大多网络支持PIL格式的数据输入。

1. [PIL一些操作](https://blog.csdn.net/dcrmg/article/details/102963336)
2. [pil_to_pytorch](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.pil_to_tensor.html)


注意：PIL只两种数据格式，0-255(uint8)，或者0-1(float)。

