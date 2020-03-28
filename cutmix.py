import copy
import numpy as np


def get_random_boundingbox(image, l_param):
    width = image.shape[0]
    height = image.shape[1]

    r_x = np.random.randint(width)
    r_y = np.random.randint(height)

    r_l = np.sqrt(1 - l_param)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)

    if r_x + r_w < width:
        bbox_x1 = r_x
        bbox_x2 = r_w
    else:
        bbox_x1 = width - r_w
        bbox_x2 = width
    if r_y + r_h < height:
        bbox_y1 = r_y
        bbox_y2 = r_h
    else:
        bbox_y1 = height - r_h
        bbox_y2 = height

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2


def cutmix(image_batch, label_batch, beta=0.5, is_debug=False):
    batch_size = len(image_batch)

    l_param = np.random.beta(beta, beta, batch_size)

    index = np.random.permutation(batch_size)
    x1, x2 = image_batch, image_batch[index]
    y1 = np.array(label_batch).astype(np.float)
    y2 = np.array(np.array(label_batch)[index]).astype(np.float)

    x = copy.deepcopy(x2)
    y = copy.deepcopy(y2)

    for i in range(batch_size):
        bx1, by1, bx2, by2 = get_random_boundingbox(x1[i], l_param[i])
        x[i][bx1:bx2, by1:by2, :] = x1[i][bx1:bx2, by1:by2, :]
        y[i] = l_param[i] * y2[i] + (1 - l_param[i]) * y1[i]

    if not is_debug:
        return x, y
    else:
        return x, y, x1, y1, x2, y2, l_param
