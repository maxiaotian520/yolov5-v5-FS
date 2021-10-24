# !git clone https://github.com/maxiaotian520/yolov5_v5-FS.git  # clone repo
# 克隆自己的github项目，方法是在自己的github地址前面加上用户名+密钥@github.com---, 这个密钥我创建的是永久不过期的，参考
# From your GitHub account, go to Settings => Developer Settings => Personal Access Token => Generate New Token (Give your password) => 
# Fillup the form => click Generate token => Copy the generated Token, it will be something like ghp_sFhFsSHhTzMDreGRLjmks4Tzuzgthdvfsrta
!git clone https://maxiaotian520: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@github.com/maxiaotian520/yolov5_v5-FS.git
%cd yolov5_v5-FS
%pip install -qr requirements.txt  # install dependencies

# Train YOLOv5s on COCO128 for 3 epochs
# 这条命令用来下载VOC 数据集  直接在cell 里run
!python train.py --img 640 --batch 16 --epochs 3 --data VOC.yaml --weights '' --cfg yolov5s.yaml

# train VOC 数据集 (打开terminal/cell)
!python train_FS.py --data VOC.yaml --cfg yolov5s.yaml --weights '' --batch-size 64 --sr 0.47 --threshold 0.01 --epochs 300




## 防止长时间训练后colab 停止session   https://www.codenong.com/57113226/
## 在您的桌面上运行此代码，然后将鼠标箭头指向任何目录上的（colabs 左侧面板 - 文件部分）目录结构，此代码将每 30 秒继续单击目录，因此它将每 30 秒扩展和收缩一次，因此您的会话不会过期重要 - 
## 你必须在你的电脑上运行这个代码
from pynput.mouse import Button, Controller 
import time
mouse = Controller()
while True:
    mouse.click(Button.left, 1)
    time.sleep(30)