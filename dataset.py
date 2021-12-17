import numpy
import cv2
import os
import re
import tiffile
from torch.utils.data import Dataset


def page2array(page):
    data = page.asarray()
    data = numpy.array(data)
    return data


def find_closest(items, size):
    result = items[0]
    diff = abs(result[0][0] - size)
    for item in items[1:]:
        t_diff = abs(item[0][0] - size)
        if t_diff < diff:
            result = item
            diff = t_diff
    return result


class LargeTifDataset(Dataset):
    def __init__(self, length, tifs, transform):
        self.tifs = numpy.array(tifs)
        self.transform = transform
        self.current = []
        self.length = length
        self.reset()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = idx % len(self.current)
        item = self.current[i]
        if self.transform is not None:
            return self.transform(item)
        return item

    def reset(self):
        self.current = []
        while True:
            idx = numpy.random.randint(0, len(self.tifs), 3)
            if len(set(idx)) == len(idx):
                break
        for i in idx:
            layers = load_tif(self.tifs[i])
            item = find_closest(layers, 500)
            img = page2array(item[1])
            self.current.append(img)


def load_tif(path):
    tif = tiffile.TiffFile(path)

    biggest = None
    b_size = 0
    result = []
    for page in tif.pages:
        size = page.size / 1e6
        if b_size < size:
            b_size = size
            biggest = page
        num = '(\d+\.?\d*)'
        parse = re.match('level={0}\smag={0}\squality={0}'.format(num), page.description)
        if parse is None:
            continue
        else:
           mag = parse.group(2)
           result.append(((size, float(mag)), page))
    result.sort()
    return result


class TinyImageNet(Dataset):
    def __init__(self, path):
        super().__init__()
        class_dirs = os.listdir(path)
        self.files = []

        for d in class_dirs:
            dir_path = os.path.join(path, d, 'images')
            for f in os.listdir(dir_path):
                self.files.append(os.path.join(dir_path, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = numpy.moveaxis(img, (0, 1, 2), (1, 2, 0))
        img = img / 255.0
        return img


