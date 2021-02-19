## MODNet - ONNX Model

This ONNX version of MODNet is provided by [@manthan3C273](https://github.com/manthan3C273) from the community.  
Please note that the PyTorch version required for this ONNX export function is higher than the official MODNet code (torch==1.7.1 is recommended).

You can try **MODNet - Image Matting Demo (ONNX version)** in [this Colab](https://colab.research.google.com/drive/1P3cWtg8fnmu9karZHYDAtmm1vj1rgA-f?usp=sharing).  
You can also download the ONNX version of the official **Image Matting Model** from [this link](https://drive.google.com/file/d/1cgycTQlYXpTh26gB9FTnthE7AvruV8hd/view?usp=sharing).

To export the ONNX version of MODNet (assuming you are currently in project root directory):
1. Download the pre-trained **Image Matting Model** from this [link](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing) and put the model into the folder `MODNet/pretrained/`.

2. Install all dependencies by:  
    ```
    pip install -r onnx/requirements.txt
    ```

3. Export the ONNX version of MODNet by: 
    ```shell
    python -m onnx.export_onnx \
        --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
        --output-path=pretrained/modnet_photographic_portrait_matting.onnx
    ```

4. Inference the ONNX model by:
    ```shell
    python -m onnx.inference_onnx \
        --image-path=$FILENAME_OF_INPUT_IMAGE$ \
        --output-path=$FILENAME_OF_OUTPUT_MATTE$ \
        --model-path=pretrained/modnet_photographic_portrait_matting.onnx
    ```
