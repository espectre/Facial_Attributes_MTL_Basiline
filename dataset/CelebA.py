from PIL import Image
import torch.utils.data as data
import numpy as np
import os


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_img(part_dir, partition):
    img = []
    with open(part_dir) as f:
        lines = f.readlines()
        for line in lines:
            pic_dir, num = line.split()
            if num == partition: 
                img.append(pic_dir)
    return img

class CelebA(data.Dataset):
    def __init__(self, part_dir, attr_dir, partition, img_dir, transform):
        self.attr = np.zeros((202600, 40))
        with open(attr_dir) as f:
            f.readline()
            f.readline()
            lines = f.readlines()
            id = 0
            for line in lines:
                vals = line.split()
                id += 1
                for j in range(40):
                    self.attr[id, j] = int(vals[j+1])
        self.img= make_img(part_dir, partition)
        self.length = len(self.img)
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, index):
        image = pil_loader(os.path.join(self.img_dir, self.img[index]))
        if self.transform is not None:
            image = self.transform(image)
        id = int(self.img[index].split('.')[0])
        return image, self.attr[id, :]


    def __len__(self):
        return self.length
