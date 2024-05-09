# SUCC官方源代码

![image-20240509105906087.png](teaser%2Fimage-20240509105906087.png)
借助场景先验，实现人群计数模型的跨领域自适应迁移


## 安装方式

```sh
pip install -r requirements.txt
```

数据集支持主流的视频人群数据集，也支持自定义输入人群视频数据




我们提供的两个预训练模型，通过百度网盘

链接: https://pan.baidu.com/s/19QVkP-nuMqN9C036qUV4ug 提取码: 6rw6 



将两个模型分别复制到如下路径

best-model_epoch-204_mae-0.0505_loss-0.1370.pth -> saliency/models/best-model_epoch-204_mae-0.0505_loss-0.1370.pth

esa_net.pth.tar -> /esr_net/weights/esa_net.pth.tar



## 运行

```sh
python main.py
```

