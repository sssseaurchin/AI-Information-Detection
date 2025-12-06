# AI-Information-Detection

## Overview
eh lol idunno

## Dependencies
Python **3.10** *(Miniconda Recommended)*

Tensorflow **< 2.11**

Pandas = **2.3.3**

Numpy = **1.26.4**


## Installation

Environment setup modified from https://www.tensorflow.org/install/pip#windows-native.

### **If installing on Windows native**
```
conda create -n tf python=3.10 numpy=1.26.4 pandas
conda activate tf
pip install "tensorflow<2.11"
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

if keras returns a could not be resolved error restart pylance by disableing then enabling the extension

more instructions to come
