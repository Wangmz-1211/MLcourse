import os
from PIL import Image
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform()
        # root = './Dataset' 总之指向/1和/2文件夹的上一层
        # transform = torchvision.transforms.ToTensor()
        # 先得到两种情况的数据路径
        train_1_path = root + '/1'
        train_2_path = root + '/2'
        # 得到两种情况的文件名列表
        train_1_list = os.listdir(train_1_path)
        train_2_list = os.listdir(train_2_path)
        # 得到两种情况的数据量
        train_1_size = len(train_1_list)
        train_2_size = len(train_2_list)
        # 构建target列表
        targets_1 = [0] * train_1_size
        targets_2 = [1] * train_2_size
        # 拼接列表
        self.imgs = train_1_list + train_2_list
        self.targets = targets_1 + targets_2

    def __getitem__(self, index):
        # 先得到target确定目录
        target = self.targets[index]
        # 声明变量名
        img_path = self.root
        # 拼接图片路径
        if target == 0:
            img_path = self.root + '/1/' + self.imgs[index]
        if target == 1:
            img_path = self.root + '/2/' + self.imgs[index]
        # 根据图片路径读取数据
        img = Image.open(img_path)
        # 转换成Tensor类型
        img = self.transform(img)
#        target = self.transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
