# 说明
代码fork from [MODNet官方代码](https://github.com/ZHKKKe/MODNet) 。本项目完善了数据准备、模型评价及模型训练相关代码
# 模型训练、评价、推理
```bash
# 1. 下载代码并进入工作目录
git clone https://github.com/actboy/MODNet
cd MODNet

# 2. 安装依赖
pip install -r src/requirements.txt

# 3. 下载并解压数据集
wget -c https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -O src/datasets/PPM-100.zip
unzip src/datasets/PPM-100.zip -d src/datasets

# 4. 训练模型
python src/trainer.py

# 5. 模型评估
python src/eval.py

# 6. 模型推理
python src/infer.py
```
