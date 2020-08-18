import os
import numpy as np
import cv2
import glob
import random


def datalist_loader(path, reg_type=None, data_type='train'):
    data_path = os.path.join(path, reg_type, data_type)

    contrasts = ['T1', 'T2', 'FL']
    gt_motion = ['gt', 'motion']

    datalist = []
    for x_or_y in gt_motion:
        for contrast in contrasts:
            datalist.append(sorted(glob.glob(data_path+'/'+contrast+'_'+x_or_y+'/*.png')))

    return datalist[0:3], datalist[3:6]


def load_batch(fname_list):
    out = []
    for fname in fname_list:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        out.append(img)
    out = np.expand_dims(np.array(out).astype('float32'), axis=-1)
    out = out/255
    return out


def train_batch_data_loader(datalist, num_contrast, shuffle=True):
    datalist_t = list(zip(*datalist))

    train_size = len(datalist_t)
    if shuffle:
        random.shuffle(datalist_t)
        datalist = list(zip(*datalist_t))

    datalist_out_y = []
    datalist_out_x = []
    for i in range(train_size):
        datalist_single_y = []
        datalist_single_x = []
        for j in range(num_contrast):
            gt_or_motion = random.randint(0, 1)*num_contrast        # 0 for gt input, 3 for motion input
            datalist_single_y.append(datalist[j][i])
            datalist_single_x.append(datalist[j+gt_or_motion][i])

        datalist_out_y.append(datalist_single_y)
        datalist_out_x.append(datalist_single_x)

    datalist_out_y = list(map(list, zip(*datalist_out_y)))
    datalist_out_x = list(map(list, zip(*datalist_out_x)))

    return datalist_out_y, datalist_out_x



def batch_data_loader(batch_datalist, num_contrast):
    '''
    Arguments:
        datalist = list
        num_contrast = The number of contrast
    '''

    batch = []
    for i in range(num_contrast):
        batch.append(load_batch(batch_datalist[i]))

    return batch