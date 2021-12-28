import cv2
import torch


def init_weights(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform(m.weight)
    if hasattr(m, 'bias'):
        m.bias.fill_(0.01)


# adapted from https://github.com/PyImageSearch/imutils
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(round(w * r)), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(round(h * r)))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# https://www.heikohoffmann.de/htmlthesis/node134.html
def iterative_mean(mean, t, x):
    return mean + 1 / (t + 1) * (x - mean)


def update_average(result, t, new):
    for k, v in new.items():
        mean = 0
        if k in result:
            mean = result[k]
        result[k] = iterative_mean(mean, t, v)
    return result


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return y_cat

