<h2 align="center">MODNet: Trimap-Free Portrait Matting in Real Time</h2>

<img src="doc/gif/homepage_demo.gif" width="100%">

<div align="center">MODNet is a model for <b>real-time</b> portrait matting with <b>only RGB image input</b>.</div>
<div align="center">MODNet是一个<b>仅需RGB图片输入</b>的<b>实时</b>人像抠图模型。</div>

<br />

<p align="center">
  <a href="#online-solution-在线方案">Online Solution (在线方案)</a> |
  <a href="#research-demo">Research Demo</a> | 
  <a href="https://arxiv.org/pdf/2011.11961.pdf">Arxiv Preprint</a> |
  <a href="https://youtu.be/PqJ3BRHX3Lc">Supplementary Video</a>
</p>

<p align="center">
  <a href="#community">Community</a> |
  <a href="#code">Code</a> |
  <a href="#ppm-benchmark">PPM Benchmark</a> | 
  <a href="#license">License</a> | 
  <a href="#acknowledgement">Acknowledgement</a> | 
  <a href="#citation">Citation</a> | 
  <a href="#contact">Contact</a>
</p>

**News:** We create a repository for our new model [MODNet-V](https://github.com/ZHKKKe/MODNet-V) that focuses on faster and better portrait video matting.  
**News:** The PPM-100 benchmark is released in [this repository](https://github.com/ZHKKKe/PPM).

---


## Online Solution (在线方案)

The online solution for portrait matting is coming!  
人像抠图在线方案发布了！ 

### Portrait Image Matting Solution (图片抠像方案)

A **Single** Model! Only **7M**! Process **2K** resolution image with a **Fast** speed on common PCs or Mobiles!  
**单个**模型！大小仅为**7M**！可以在普通PC或移动设备上**快速**处理具有**2K**分辨率的图像！ 

Now you can try our **portrait image matting** online via [this website](https://sight-x.cn/portrait_matting).  
现在，您可以通过[此网站](https://sight-x.cn/portrait_matting)在线使用我们的**图片抠像**功能。  

<img src="doc/gif/commercial_image_matting_website.gif" width='100%'>

<!-- You can also scan the QR code below to try the WeChat Mini-Program based on our model.  
您也可以扫描下方二维码尝试基于我们模型的微信小程序。 -->

<!-- Here are two example videos processed (frame independently) via our **portrait image matting** model:  
我们**图片抠像**模型逐帧单独处理的两个示例视频: 

<img src="doc/gif/commercial_image_matting_model_result.gif" width='100%'> -->


<!-- ### Portrait Video Matting Solution (视频抠像方案)

Stay tuned.  
敬请期待。
 -->

<!-- --- -->


## Research Demo

All the models behind the following demos are trained on the datasets mentioned in [our paper](https://arxiv.org/pdf/2011.11961.pdf). 

### Portrait Image Matting
We provide an [online Colab demo](https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing) for portrait image matting.  
It allows you to upload portrait images and predict/visualize/download the alpha mattes. 

<!-- <img src="doc/gif/image_matting_demo.gif" width='40%'> -->

### Portrait Video Matting
We provide two real-time portrait video matting demos based on WebCam. When using the demo, you can move the WebCam around at will. 
If you have an Ubuntu system, we recommend you to try the [offline demo](demo/video_matting/webcam) to get a higher *fps*. Otherwise, you can access the [online Colab demo](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing).  
We also provide an [offline demo](demo/video_matting/custom) that allows you to process custom videos.

<!-- <img src="doc/gif/video_matting_demo.gif" width='60%'> -->


## Community

We share some cool applications/extentions of MODNet built by the community.

- **WebGUI for Portrait Image Matting**  
You can try [this WebGUI](https://www.gradio.app/hub/aliabd/modnet) (hosted on [Gradio](https://www.gradio.app/)) for portrait image matting from your browser without code! 

- **Colab Demo of Bokeh (Blur Background)**  
You can try [this Colab demo](https://colab.research.google.com/github/eyaler/avatars4all/blob/master/yarok.ipynb) (built by [@eyaler](https://github.com/eyaler)) to blur the backgroud based on MODNet!

- **ONNX Version of MODNet**  
You can convert the pre-trained MODNet to an ONNX model by using [this code](onnx) (provided by [@manthan3C273](https://github.com/manthan3C273)). You can also try [this Colab demo](https://colab.research.google.com/drive/1P3cWtg8fnmu9karZHYDAtmm1vj1rgA-f?usp=sharing) for MODNet image matting (ONNX version).

- **TorchScript Version of MODNet**  
You can convert the pre-trained MODNet to an TorchScript model by using [this code](torchscript) (provided by [@yarkable](https://github.com/yarkable)). 

- **TensorRT Version of MODNet**  
You can access [this Github repository](https://github.com/jkjung-avt/tensorrt_demos) to try the TensorRT version of MODNet (provided by [@jkjung-avt](https://github.com/jkjung-avt)).


There are some resources about MODNet from the community.
- [Video from What's AI YouTube Channel](https://youtu.be/rUo0wuVyefU)
- [Article from Louis Bouchard's Blog](https://www.louisbouchard.ai/remove-background/)


## Code
We provide the [code](src/trainer.py) of MODNet training iteration, including:
- **Supervised Training**: Train MODNet on a labeled matting dataset
- **SOC Adaptation**: Adapt a trained MODNet to an unlabeled dataset

In the code comments, we provide examples for using the functions.  


## PPM Benchmark
The PPM benchmark is released in a separate repository [PPM](https://github.com/ZHKKKe/PPM).  


## License
All resources in this repository (code, models, demos, *etc.*) are released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.  
The license will be changed to allow commercial use after our paper is accepted.


## Acknowledgement  
- We thank  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[@eyaler](https://github.com/eyaler), [@manthan3C273](https://github.com/manthan3C273),  [@yarkable](https://github.com/yarkable), [@jkjung-avt](https://github.com/jkjung-avt),  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[the Gradio team](https://github.com/gradio-app/gradio), [What's AI YouTube Channel](https://www.youtube.com/channel/UCUzGQrN-lyyc0BWTYoJM_Sg), [Louis Bouchard's Blog](https://www.louisbouchard.ai),  
for their contributions to this repository or their cool applications/extentions/resources of MODNet.


## Citation
If this work helps your research, please consider to cite:

```bibtex
@article{MODNet,
  author = {Zhanghan Ke and Kaican Li and Yurou Zhou and Qiuhua Wu and Xiangyu Mao and Qiong Yan and Rynson W.H. Lau},
  title = {Is a Green Screen Really Necessary for Real-Time Portrait Matting?},
  journal={ArXiv},
  volume={abs/2011.11961},
  year = {2020},
}
```


## Contact
This repository is currently maintained by Zhanghan Ke ([@ZHKKKe](https://github.com/ZHKKKe)).  
For questions, please contact `kezhanghan@outlook.com`.
