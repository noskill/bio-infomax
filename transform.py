import numpy


def random_crop(image, output_size, return_pos=False):
    argmin = numpy.argmin(image.shape)
    if argmin == 0:
        image = image.transpose(1, 2, 0)
    h, w = image.shape[:2]
    new_h, new_w = output_size
    diff_h = h - new_h
    diff_w = w - new_w
    top = 0 if (diff_h <= 0) else numpy.random.randint(0, diff_h)
    left = 0 if (diff_w <= 0) else numpy.random.randint(0, diff_w)
    image = image[top: top + new_h, left: left + new_w]
    if argmin == 0:
        image = image.transpose(2, 0, 1)
    if return_pos:
        return image, (top, left)
    return image


class RandomCropTransform:
    def __init__(self, size, beta=0):
        self.size = size
        self.beta = beta

    def __call__(self, data, return_pos=False):
        if self.beta:
            size = self.size + int(numpy.random.random() * self.beta * 2) - self.beta
        else:
            size = self.size
        img, pos = random_crop(data, (size, size), return_pos=True)
        if return_pos:
            return img, pos
        return img

