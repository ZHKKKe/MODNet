## MODNet - WebCam-Based Portrait Video Matting Demo
This is a MODNet portrait video matting demo based on WebCam. It will call your local WebCam and display the matting results in real time. The demo can run under CPU or GPU.

### 1. Requirements
The basic requirements for this demo are:
- Ubuntu System
- WebCam
- Python 3+

**NOTE**: If your device does not satisfy the above conditions, please try our [online Colab demo](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing).


### 2. Introduction
We use ~400 unlabeled video clips (divided into ~50,000 frames) downloaded from the internet to perform SOC to adapt MODNet to the video domain. **Nonetheless, due to insufficient labeled training data (~3k labeled foregrounds), our model may still make errors in portrait semantics estimation under challenging scenes.** Besides, this demo does not currently support the OFD trick, which will be provided soon.

For a better experience, please:

*   make sure the portrait and background are distinguishable, <i>i.e.</i>, are not similar
*   run in soft and bright ambient lighting
*   do not be too close or too far from the WebCam
*   do not move too fast

### 3. Run Demo
We recommend creating a new conda virtual environment to run this demo, as follow:

1. Clone the MODNet repository:
    ```
    git clone https://github.com/ZHKKKe/MODNet.git
    cd MODNet
    ```

2. Download the pre-trained model from this [link](https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/view?usp=sharing) and put it into the folder `MODNet/pretrained/`.


3. Create a conda virtual environment named `modnet` (if it doesn't exist) and activate it. Here we use `python=3.6` as an example:
     ```
    conda create -n modnet python=3.6
    source activate modnet
    ```

4. Install the required python dependencies (please make sure your CUDA version is supported by the PyTorch version installed):
    ```
    pip install -r demo/video_matting/webcam/requirements.txt
    ```

5. Execute the main code:
    ```
    python -m demo.video_matting.webcam.run
    ```

### 4. Acknowledgement
We thank [@tkianai](https://github.com/tkianai) and [@mazhar004](https://github.com/mazhar004) for their contributions to making this demo available for CPU use.
