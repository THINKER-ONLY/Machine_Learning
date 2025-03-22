@echo off
REM 创建新环境并安装核心依赖
call conda create -n MC python=3.10.12 -y
call conda activate MC

REM 安装CUDA工具链（GPU用户必选）
call conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.6.0 -y

REM 安装深度学习框架
call pip install tensorflow==2.10.1
call conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

REM 安装科学计算全家桶
call pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1 jupyterlab==3.6.1 

REM 安装扩展工具包
call pip install opencv-python==4.7.0.72 xgboost==1.7.5 lightgbm==3.3.5 tqdm==4.65.0 seaborn==0.12.2

REM 环境验证
echo 正在验证GPU支持...
python -c "import tensorflow as tf; print('\nTensorFlow GPU可用' if tf.config.list_physical_devices('GPU') else '警告：未检测到GPU')"
python -c "import torch; print('PyTorch GPU可用' if torch.cuda.is_available() else '警告：PyTorch未检测到GPU')"

echo 环境配置完成！使用 conda activate dl_win 激活环境
pause