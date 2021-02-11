# Inference with onnxruntime

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

