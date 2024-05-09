# Scene-adaptive Unsupervised Crowd Counting for Video Surveillance 


![image-20240509105906087.png](teaser%2Fimage-20240509105906087.png)
Implement cross-domain adaptive migration of crowd counting models with the help of scene priors


## Installation method

```sh
pip install -r requirements.txt
```

>The data set supports mainstream video crowd data sets and also supports custom input of crowd video data.



## Pretrained model weights

Download
>The two pre-trained models we provide are available through Baidu Netdisk

>Link: https://pan.baidu.com/s/19QVkP-nuMqN9C036qUV4ug 
>Extraction code: 6rw6


### Run path
Copy the two models to the following paths respectively:

>best-model_epoch-204_mae-0.0505_loss-0.1370.pth -> saliency/models/best-model_epoch-204_mae-0.0505_loss-0.1370.pth

>esa_net.pth.tar -> /esr_net/weights/esa_net.pth.tar



## Run

```sh
python main.py
```

