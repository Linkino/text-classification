# text-classification
本代码库包含本人所用情感分类与新闻主题分类模型，模型部署等代码

注意：

1.本案例包含情感分析与新闻主题分类模型，其中其中VKmodel—title.py是text-cnn改进情感分类模型得源码，VKmodel—BIlstm-ATT.py是BILSTM+ATT情感分类模型源码，VKmodel-theme-data.py是主题分类源码，所有模型数据都是从数据库中读取的。

2.情感分析所用数据集数量10万条，使用预训练词向量嵌入的方式进行模型训练，其中BILSTM+attention指标如下图所示： 

![图1 classreport](https://github.com/yanhan19940405/text-classification/blob/master/image/bilstmatt.png)

我结合text-cnn基础上进行模型改进，模型结果好于BILSTM_ATT结构模型，指标如下图所示： 

![图2 classreport](https://github.com/yanhan19940405/text-classification/blob/master/image/1.png)

模型结构如下图所示，训练过程中模型loss与acc曲线如下所示：  

![图3 structure](https://github.com/yanhan19940405/text-classification/blob/master/image/structure.png)

![图3.1 loss](https://github.com/yanhan19940405/text-classification/blob/master/image/loss.png)

![图3.2 acc](https://github.com/yanhan19940405/text-classification/blob/master/image/acc.png)
综上，模型已经排除了过拟合因素影响。
3.新闻主题分类数据总量为40w，使用模型也为基于text-cnn改进模型，模型效果如下图所示，分类为7类，结果待后需进一步优化：  

![图4 classreport](https://github.com/yanhan19940405/text-classification/blob/master/image/theme.png)

4.由于github容量大小限制，本代码库删除W2V预训练词向量，本地数据库相关配置项，以及所生成得模型文件。

5.上述所有分类报告皆在经过与训练集，统一数据生成规则得到得测试集所得出，本例还使用了字词联合表征得处理方式进行优化。  

6.model_server.py即为模型部署代码，通过flask部署模型文件于应用层，然后通过localhost:5000?context="新闻文本"&&title="新闻标题 "进行部署模型预测即可。
