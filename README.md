<h2 align="center">MODNet: Is a Green Screen Really Necessary for Real-Time Portrait Matting?</h2>

<img src="doc/gif/homepage_demo.gif" width="100%">

<p align="center">
  <a href="https://arxiv.org/pdf/2011.11961.pdf">Arxiv Preprint</a> |
  <a href="https://youtu.be/PqJ3BRHX3Lc">Supplementary Video</a>
</p>

<p align="center">
WebCam Video Demo [<a href="demo/video_matting/webcam">Offline</a>][<a href="https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing">Colab</a>] | Custom Video Demo [<a href="demo/video_matting/custom">Offline</a>] |
  Image Demo [<a href="https://gradio.app/g/modnet">WebGUI</a>][<a href="https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing">Colab</a>]
</p>

<div align="center">This is the official project of our paper <b>Is a Green Screen Really Necessary for Real-Time Portrait Matting?</b></div>
<div align="center">MODNet is a <b>trimap-free</b> model for portrait matting in <b>real time</b> under <b>changing scenes</b>.</div>


---


## News
- [Jan&nbsp; 28 2021] Release the [code](src/trainer.py) of MODNet training iteration.
- [Dec 25 2020] ***Merry Christmas!*** :christmas_tree: Release Custom Video Matting Demo [[Offline](demo/video_matting/custom)] for user videos.
- [Dec 10 2020] Release WebCam Video Matting Demo [[Offline](demo/video_matting/webcam)][[Colab](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing)] and Image Matting Demo [[Colab](https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing)].
- [Nov 24 2020] Release [Arxiv Preprint](https://arxiv.org/pdf/2011.11961.pdf) and [Supplementary Video](https://youtu.be/PqJ3BRHX3Lc).


## Demos

### Video Matting
We provide two real-time portrait video matting demos based on WebCam. When using the demo, you can move the WebCam around at will. 
If you have an Ubuntu system, we recommend you to try the [offline demo](demo/video_matting/webcam) to get a higher *fps*. Otherwise, you can access the [online Colab demo](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing).  
We also provide an [offline demo](demo/video_matting/custom) that allows you to process custom videos.

<img src="doc/gif/video_matting_demo.gif" width='60%'>


### Image Matting
We provide an [online Colab demo](https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing) for portrait image matting.  
It allows you to upload portrait images and predict/visualize/download the alpha mattes. 

<img src="doc/gif/image_matting_demo.gif" width='40%'>


### Community
Here we share some cool applications of MODNet built by the community.

- **WebGUI for Image Matting**  
You can try [this WebGUI](https://gradio.app/g/modnet) (hosted on [Gradio](https://www.gradio.app/)) for portrait matting from your browser without any code! 
<!-- <img src="https://i.ibb.co/9gLxFXF/modnet.gif" width='40%'> -->

- **Colab Demo of Bokeh (Blur Background)**  
You can try [this Colab demo](https://colab.research.google.com/github/eyaler/avatars4all/blob/master/yarok.ipynb) (built by [@eyaler](https://github.com/eyaler)) to blur the backgroud based on MODNet!


## Code
We provide the [code](src/trainer.py) of MODNet training iteration, including:
- **Supervised Training**: Train MODNet on a labeled matting dataset
- **SOC Adaptation**: Adapt a trained MODNet to an unlabeled dataset

In the function comments, we provide examples of how to call the function.  


## TODO
- Release the code of One-Frame Delay (OFD)
- Release PPM-100 validation benchmark (scheduled in **Feb 2021**)  
**NOTE**: PPM-100 is a **validation set**. Our training set will not be published


## License
This project (code, pre-trained models, demos, *etc.*) is released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.


## Acknowledgement
- We thank [City University of Hong Kong](https://www.cityu.edu.hk/) and [SenseTime](https://www.sensetime.com/) for their support to this project.  
- We thank  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[the Gradio team](https://github.com/gradio-app/gradio), [@eyaler](https://github.com/eyaler),  
for their cool applications based on MODNet.


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
This project is currently maintained by Zhanghan Ke ([@ZHKKKe](https://github.com/ZHKKKe)).  
If you have any questions, please feel free to contact `kezhanghan@outlook.com`.
