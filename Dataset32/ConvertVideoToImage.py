import cv2
import os

# #####################################################
# 从当前文件夹中读取视频文件 video_address（23行），逐帧读取  #
# 在当前文件夹中创建./result文件夹，并把压缩后的图像放进去     #
# #####################################################

# image：要保存的图片
# addr：图片的地址和名称信息
# num图片名称的后缀，使用int类型来计数

def save_image(image, name, num):
    address = './result/' + name + '_' + str(num) + '.jpg'
    cv2.imwrite(address, image)


#   初始化变量
i = 0  # 帧计数
j = 0  # 图片计数
timeF = 1  # 每隔1帧保存一张图片
size = (32, 32)  # 压缩后图片的尺寸
video_address = 'WIN_20221012_10_16_07_Pro.mp4'  # 视频文件
#   读取视频文件
video = cv2.VideoCapture(video_address)
#   读帧
success, frame = video.read()
if not os.path.exists('./result'):
    os.mkdir('result')
while success:
    i = i + 1
    if i % timeF == 0:
        j = i + 1
        frame_zip = cv2.resize(frame, size)
        save_image(frame_zip, 'image', j)
        print('save image:', j)
    success, frame = video.read()

'''
版权声明：本文为CSDN博主「居然.org」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_44975659/article/details/125025460    #视频转图
原文链接：https://blog.csdn.net/hhaowang/article/details/87354146        #压缩
'''
