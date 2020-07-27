![image-20200727222210368](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727222210368.png)

## 前言

最近参加了一下中国平安的**2020中国大学生保险数字科技挑战赛**的数字赛道，因为前两天刚厦门答辩完所以现在可以开源了。

我们队是**硅谷小镇队**，最后是**全国前十**，**华南赛区第二**的成绩。

最后分数是 **线下：40.36  线上：41.77**

代码开源地址：



## 赛题理解

赛题是预测未来10天每一个userid的Top3点击产品，是一个推荐保险的比赛，我们队的主要做法就是特征工程 + lgb二分类。

<img src="https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727222827341.png" alt="image-20200727222827341" style="zoom:50%;" />

## 数据分析

![image-20200727222943649](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727222943649.png)

其实从上图的历史前三产品作为未来十天的推荐产品就已经36分了，已经好像在榜上比较前的位置了。所以这样label分布极其不均匀的，我们应该对label更加在意才对。

## 特征工程

### 基础特征

我们基本的特征其实真的不多，把能删的都删了，地理位置，wifi类型全都不要了，只留了一些购买过产品的数量，产品类型的数量。

![image-20200727223419151](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727223419151.png)

### 产品关联

![image-20200727223744670](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727223744670.png)

其实没什么特别的，就是前20天购买过的产品的id列表one-hot编码了一遍，有就是1，没有就是0，形成的关联表concat到userid主表中，这是一个很自然的做法，充分挖掘label的特征。

### 规则关联

![image-20200727223959233](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727223959233.png)

针对title我们可以看到买过的产品的中文，对中文的语义特征，我们可以直接的形成一些规则，比如**他买过萌宠的产品就证明他养宠物**，**她买过女性类的产品证明她是女性**，这种规则的方法其实在很多比赛中很常见，大家可以记住一下。

## word2vec

![image-20200727224415865](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727224415865.png)

我们对点击过的浏览过的产品按时间顺序做成语料库训练word2vec，最后可以把浏览记录生成50维的特征。（这个对我们的模型其实帮助不大，可能我的用法不对）



## 转化为二分类

这是一个点击率预测的问题，我们直接转化为二分类，就是把他买过的产品后面加一维label，把产品给onehot了，一共62维产品+1维label。

但是这个都是买的label=1，不买的label=0，问题是这样在线上内存不够，如果把每个用户买过的和没买的都作一行，每个用户就有62行。所以我们采用了暴力的方法，公众号回复【2020平安数字挑战赛】即可获得数据集和ppt。

## 模型调优

![image-20200727225336206](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727225336206.png)

调参好像我用了贝叶斯调优，反正最后特征维度不大，我也没想着会进复赛，最后成绩居然还不错，建议大家可以按照我这样过一遍，数据挖掘的比赛大概都这样。

## 总结

我们就是去旅游的，所以我们去答辩的时候没啥压力，就是把做的东西说了说，本来卖保险的嘛都得穿正装，我们队都没穿（哪有程序员穿正装？），就是去交交朋友，他们包车票住宿费，整个流程挺好的，最后我们多留一晚，鼓浪屿也挺好挺幽静的，我后面玩的时候住的民宿超好，三个人190而且还给我们两间房，也就是厦门3日游玩人均只用了差不多200元...hhhhh

最后放张去**鼓浪屿**的旅游照：

![image-20200727230626522](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/image-20200727230626522.png)

**跪求大家点击关注，在看。**

![媒体120207271832111](https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master//pingan.assets/%E5%AA%92%E4%BD%93120207271832111.gif)