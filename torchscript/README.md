## MODNet - TorchScript Model

This TorchScript version of MODNet is provided by [@yarkable](https://github.com/yarkable) from the community.  
Please note that the PyTorch version required for this TorchScript export function is higher than the official MODNet code (torch>=1.2.0).

You can also download the TorchScript version of the official **Image Matting Model** from [this link](https://pan.baidu.com/s/1kOmmmbG7lSZiSmDdE7CaRw) with the exextraction code `dm9e`.

To export the TorchScript version of MODNet (assuming you are currently in project root directory):
1. Download the pre-trained **Image Matting Model** from this [link](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing) and put the model into the folder `MODNet/pretrained/`.

2. Ensure your PyTorch version >= 1.2.0.

3. Export the TorchScript version of MODNet by: 
    ```shell
    python -m torchscript.export_torchscript \
        --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
        --output-path=pretrained/modnet_photographic_portrait_matting.torchscript
    ```
