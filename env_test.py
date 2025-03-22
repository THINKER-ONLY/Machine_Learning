import tensorflow as tf
import torch
import sys

print(f"Python版本: {sys.version}")
print(f"TensorFlow版本: {tf.__version__}, GPU可用: {bool(tf.config.list_physical_devices('GPU'))}")
print(f"PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA Toolkit版本: {torch.version.cuda} (PyTorch检测)")

# 检查关键库版本
import numpy, pandas, sklearn
print(f"\nNumPy版本: {numpy.__version__}")
print(f"Pandas版本: {pandas.__version__}")
print(f"Scikit-learn版本: {sklearn.__version__}")