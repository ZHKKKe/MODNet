## MODNet - Custom Portrait Video Matting Demo
This is a MODNet portrait video matting demo that allows you to process custom videos.

### 1. Requirements
The basic requirements for this demo are:
- Ubuntu System
- Python 3+


### 2. Introduction
We use ~400 unlabeled video clips (divided into ~50,000 frames) downloaded from the internet to perform SOC to adapt MODNet to the video domain. **Nonetheless, due to insufficient labeled training data (~3k labeled foregrounds), our model may still make errors in portrait semantics estimation under challenging scenes.** Besides, this demo does not currently support the OFD trick.


For a better experience, please make sure your videos satisfy:

*   the portrait and background are distinguishable, <i>i.e.</i>, are not similar
*   captured in soft and bright ambient lighting
*   the contents do not move too fast

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
    pip install -r demo/video_matting/custom/requirements.txt
    ```

5. Execute the main code:
    ```
    python -m demo.video_matting.custom.run --video YOUR_VIDEO_PATH
    ```
    where `YOUR_VIDEO_PATH` is the specific path of your video.  
    There are some optional arguments:
     - `--result-type (default=fg)` : fg - save the alpha matte; fg - save the foreground
     - `--fps (default=30)` : fps of the result video
