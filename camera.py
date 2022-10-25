import cv2
import numpy
from pynput.keyboard import Key, Controller
import torch
from torchvision.transforms import ToTensor
from model import *


# #####################################################
# 运行本文件前，需要确认：
#   modelpath 模型路径
#   tick      每秒判定帧数
#   size      图像压缩大小，与模型一致（32*32）
# 功能：
#   从 modelpath 读取模型，并用OpenCV调用电脑摄像头
#   每秒进行 tick 次判定，输出两种结果，
#       随便写了一种情况，如果反了改一下 48行 符号
#   如果有人，就切屏，持续有人期间屏蔽切屏功能，
#       如果过于灵敏可以试试连续多次判定单人再还原标志位futari
#   按q退出
# #####################################################

def load_model(model_path):
    myModel = Model()
    myModel.load_state_dict(torch.load(model_path).get('state_dict'))
    myModel.eval()
    return myModel


def run_model(frame, model=None):
    return model(frame)


def tab(keyboard):
    keyboard.press(Key.alt)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(Key.alt)


def demo(tick, size, model_path, threshold):  # 每秒取样数
    # reference: https://www.linuxprobe.com/python-linux-two.html
    print('开始')
    keyboard = Controller()
    trans = ToTensor()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 电脑自身摄像头
    gap = 60 / tick  # 取样间隔
    i = 0
    myModel = load_model(model_path)
    futari = 0
    while True:
        i += 1
        if i == gap:  # 定时
            reg, frame = cap.read()
            frame = cv2.resize(frame, size)
            frame = cv2.flip(frame, 1)  # 图片左右调换
            cv2.imshow('window', frame)
            # <logic>
            frame = trans(frame)
            frame = torch.reshape(frame, (1, 3, 32, 32))
            result = run_model(frame, myModel)  # 将frame扔到模型里，输出分类结果
            result = result[0]
            print(result[0].item(), result[1].item())
            # 以下代码思路： 鉴于灵敏度有限，需要消除抖动
            # futari标志位取值范围为0 ~ 2*threshold
            # 当识别为两个人时 futari ++,通过threshold时切屏 ，==2*threshold则不++
            # 识别为一个人时 futari--，==0则不--
            if result[0].item() < result[1].item():
                print('二人\t', '●' * futari)

                if futari < 2 * threshold:
                    futari += 1

                if futari == threshold:
                    tab(keyboard)

            else:
                print('一人\t', '●' * futari)

                if futari > 0:
                    futari -= 1

            # </logic>
            i = 0  # 清零
            # 按Q停止
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # 释放资源
    cap.release()


tick = 10
size = (32, 32)
modelpath = './model_best.pth.tar'
threshold = 10  # 灵敏度

demo(tick, size, modelpath, threshold)
cv2.destroyAllWindows()
