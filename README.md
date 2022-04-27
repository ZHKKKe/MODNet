<h2 align="center">MODNet: Trimap-Free Portrait Matting in Real Time</h2>

<div align="center"><i>MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition (AAAI 2022)</i></div>

<br />

<img src="doc/gif/homepage_demo.gif" width="100%">

<div align="center">MODNet is a model for <b>real-time</b> portrait matting with <b>only RGB image input</b></div>
<div align="center">MODNet是一个<b>仅需RGB图片输入</b>的<b>实时</b>人像抠图模型</div>

<br />

<p align="center">
  <a href="#online-application-在线应用">Online Application (在线应用)</a> |
  <a href="#research-demo">Research Demo</a> |
  <a href="https://arxiv.org/pdf/2011.11961.pdf">AAAI 2022 Paper</a> |
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

---


## Online Application (在线应用)

A **Single** model! Only **7M**! Process **2K** resolution image with a **Fast** speed on common PCs or Mobiles! **Beter** than research demos!  
Please try online portrait image matting via [this website](https://sight-x.cn/portrait_matting) or on [my personal homepage](https://zhke.io/#/?modnet_demo)!    

**单个**模型！大小仅为**7M**！可以在普通PC或移动设备上**快速**处理具有**2K**分辨率的图像！效果比研究示例**更好**！  
请通过[此网站](https://sight-x.cn/portrait_matting)或[我的主页](https://zhke.io/#/?modnet_demo)在线尝试图片抠像！


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

- **Docker Container for MODnet**  
You can access [this Github repository](https://github.com/nahidalam/modnet_docker) for a containerized version of MODNet with the Docker environment (provided by [@nahidalam](https://github.com/nahidalam)).


There are some resources about MODNet from the community.
- [Video from What's AI YouTube Channel](https://youtu.be/rUo0wuVyefU)
- [Article from Louis Bouchard's Blog](https://www.louisbouchard.ai/remove-background/)


## Code
We provide the [code](src/trainer.py) of MODNet training iteration, including:
- **Supervised Training**: Train MODNet on a labeled matting dataset
- **SOC Adaptation**: Adapt a trained MODNet to an unlabeled dataset

In code comments, we provide examples for using the functions.  


## PPM Benchmark
The PPM benchmark is released in a separate repository [PPM](https://github.com/ZHKKKe/PPM).  


## License
The code, models, and demos in this repository (excluding GIF files under the folder `doc/gif`) are released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.  


## Acknowledgement  
- We thank  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[@yzhou0919](https://github.com/yzhou0919), [@eyaler](https://github.com/eyaler), [@manthan3C273](https://github.com/manthan3C273),  [@yarkable](https://github.com/yarkable), [@jkjung-avt](https://github.com/jkjung-avt),  [@manzke](https://github.com/manzke),  [@nahidalam](https://github.com/nahidalam),  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[the Gradio team](https://github.com/gradio-app/gradio), [What's AI YouTube Channel](https://www.youtube.com/channel/UCUzGQrN-lyyc0BWTYoJM_Sg), [Louis Bouchard's Blog](https://www.louisbouchard.ai),  
for their contributions to this repository or their cool applications/extentions/resources of MODNet.


## Citation
If this work helps your research, please consider to cite:

```bibtex
@InProceedings{MODNet,
  author = {Zhanghan Ke and Jiayu Sun and Kaican Li and Qiong Yan and Rynson W.H. Lau},
  title = {MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition},
  booktitle = {AAAI},
  year = {2022},
}
```


## Contact
This repository is maintained by Zhanghan Ke ([@ZHKKKe](https://github.com/ZHKKKe)).  
For questions, please contact `kezhanghan@outlook.com`.

<img src="doc/gif/commercial_image_matting_model_result.gif" width='100%'>
