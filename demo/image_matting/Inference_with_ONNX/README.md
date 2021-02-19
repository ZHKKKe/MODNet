# Inference with onnxruntime

Please try MODNet image matting onnx-inference demo with [Colab Notebook](https://colab.research.google.com/drive/1P3cWtg8fnmu9karZHYDAtmm1vj1rgA-f?usp=sharing) 

Download [modnet.onnx](https://drive.google.com/file/d/1cgycTQlYXpTh26gB9FTnthE7AvruV8hd/view?usp=sharing)

### 1. Export onnx model

Run the following command:
```shell
python export_modnet_onnx.py \
    --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
    --output-path=modnet.onnx
```


### 2. Inference 

Run the following command:
```shell
python inference_onnx.py \
    --image-path=PATH_TO_IMAGE \
    --output-path=matte.png \
    --model-path=modnet.onnx
```

