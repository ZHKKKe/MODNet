<h2 align="center">MODNet: Is a Green Screen Really Necessary for Real-Time Portrait Matting?</h2>

<img src="doc/gif/homepage_demo.gif" width="100%">

<p align="center">
  <a href="https://arxiv.org/pdf/2011.11961.pdf">Arxiv Preprint</a> |
  <a href="https://youtu.be/PqJ3BRHX3Lc">Supplementary Video</a> |
  <a href="https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing">Video Matting Demo</a> :fire: |
  <a href="https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing">Image Matting Demo</a> :fire: 
</p>

<div align="center">This is the official project of our paper <b>Is a Green Screen Really Necessary for Real-Time Portrait Matting?</b></div>
<div align="center">MODNet is a <b>trimap-free</b> model for portrait matting in <b>real time</b> (on a single GPU).</div>


---


## News
- [Dec 10 2020] Release [Video Matting Demo](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing) and [Image Matting Demo](https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing).
- [Nov 24 2020] Release [Arxiv Preprint](https://arxiv.org/pdf/2011.11961.pdf) and [Supplementary Video](https://youtu.be/PqJ3BRHX3Lc).


## Video Matting Demo 
We provide two real-time portrait video matting demos based on WebCam.   
If you have an Ubuntu system, we recommend you to try the [offline demo](demo/video_matting) to get a higher *fps*. Otherwise, you can access the [online Colab demo](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing).


## Image Matting Demo
We provide an [online Colab demo](https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing) for portrait image matting.  
It allows you to upload portrait images and predict/visualize/download the alpha mattes. 

<img src="doc/gif/image_matting_demo.gif">


## TO DO
- Release training code (scheduled in **Jan. 2021**)
- Release PPM-100 validation benchmark (scheduled in **Feb. 2021**)


## License
This project is released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.


## Acknowledgement
We thank [City University of Hong Kong](https://www.cityu.edu.hk/) and [SenseTime](https://www.sensetime.com/) for their support to this project.


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
