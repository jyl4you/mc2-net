import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

from model import MC_Net
from dataset import datalist_loader, batch_data_loader
from utils import test_ssim, test_nmi, test_nrmse, save_image

tf.random.set_seed(22)
np.random.seed(22)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_integer('batch_size', 1,
                                  'Batch size (Default: 1)')
tf.compat.v1.flags.DEFINE_integer('image_size', 256,
                                  'Image size (size x size) (Default: 256)')
tf.compat.v1.flags.DEFINE_string('load_weight_name', 'weight_min_val_loss.h5',
                                 'Load weight of given name (Default: weight_min_val_loss.h5)')
tf.compat.v1.flags.DEFINE_integer('num_contrast', 3,
                                  'Number of contrasts of MR images (Default: 3)')
tf.compat.v1.flags.DEFINE_integer('num_filter', 16,
                                  'Number of filters in the first layer of the encoder (Default: 16)')
tf.compat.v1.flags.DEFINE_integer('num_res_block', 9,
                                  'Number of residual blocks (Default: 9)')
tf.compat.v1.flags.DEFINE_string('path_data', './data/',
                                 'Data load path (Default: ./data/')
tf.compat.v1.flags.DEFINE_string('path_save', './test/',
                                 'Image save path (Default: ./test/')
tf.compat.v1.flags.DEFINE_string('path_weight', './weight/',
                                 'Weight load path (Default: ./weight/')
tf.compat.v1.flags.DEFINE_string('reg_type', 'NMI',
                                 'Registration type of input images (No, Network, or NMI) (Default: NMI)')

assert tf.__version__.startswith('2.')
print('Tensorflow version: ', tf.__version__)
tf.random.set_seed(22)
np.random.seed(22)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def test():
    model = MC_Net(img_size=FLAGS.image_size,
                   num_filter=FLAGS.num_filter,
                   num_contrast=FLAGS.num_contrast,
                   num_res_block=FLAGS.num_res_block)

    input_shape = [(None, FLAGS.image_size, FLAGS.image_size, 1)]
    model.build(input_shape=input_shape * FLAGS.num_contrast)

    model.load_weights(FLAGS.path_weight + FLAGS.load_weight_name)
    print('Model building completed!')

    y_test_datalist, x_test_datalist = datalist_loader(FLAGS.path_data, FLAGS.reg_type, 'test')
    x_test = batch_data_loader(x_test_datalist, FLAGS.num_contrast)
    y_test = batch_data_loader(y_test_datalist, FLAGS.num_contrast)
    print('Data loading completed!')

    p_test = model.predict(x_test, batch_size=FLAGS.batch_size)
    print('Prediction completed!')

    x_ssim_T1, p_ssim_T1 = test_ssim(x_test[0], y_test[0], p_test[0])
    x_ssim_T2, p_ssim_T2 = test_ssim(x_test[1], y_test[1], p_test[1])
    x_ssim_FL, p_ssim_FL = test_ssim(x_test[2], y_test[2], p_test[2])

    print(' c | x_ssim | p_ssim')
    print(f'T1 | {x_ssim_T1:.4f} | {p_ssim_T1:.4f}')
    print(f'T2 | {x_ssim_T2:.4f} | {p_ssim_T2:.4f}')
    print(f'FL | {x_ssim_FL:.4f} | {p_ssim_FL:.4f}')

    x_nmi_T1, p_nmi_T1 = test_nmi(x_test[0], y_test[0], p_test[0])
    x_nmi_T2, p_nmi_T2 = test_nmi(x_test[1], y_test[1], p_test[1])
    x_nmi_FL, p_nmi_FL = test_nmi(x_test[2], y_test[2], p_test[2])

    print(' c | x_nmi  | p_nmi ')
    print(f'T1 | {x_nmi_T1:.4f} | {p_nmi_T1:.4f}')
    print(f'T2 | {x_nmi_T2:.4f} | {p_nmi_T2:.4f}')
    print(f'FL | {x_nmi_FL:.4f} | {p_nmi_FL:.4f}')

    x_nrmse_T1, p_nrmse_T1 = test_nrmse(x_test[0], y_test[0], p_test[0])
    x_nrmse_T2, p_nrmse_T2 = test_nrmse(x_test[1], y_test[1], p_test[1])
    x_nrmse_FL, p_nrmse_FL = test_nrmse(x_test[2], y_test[2], p_test[2])

    print(' c | x_nrmse | p_nrmse')
    print(f'T1 | {x_nrmse_T1:.4f}  | {p_nrmse_T1:.4f}')
    print(f'T2 | {x_nrmse_T2:.4f}  | {p_nrmse_T2:.4f}')
    print(f'FL | {x_nrmse_FL:.4f}  | {p_nrmse_FL:.4f}')

    os.makedirs(FLAGS.path_save, exist_ok=True)
    for i in range(p_test[0].shape[0]):
        save_image(f'{FLAGS.path_save}/T1_pred_{i+1:04d}.png', p_test[0][i])
        save_image(f'{FLAGS.path_save}/T2_pred_{i+1:04d}.png', p_test[1][i])
        save_image(f'{FLAGS.path_save}/FL_pred_{i+1:04d}.png', p_test[2][i])
    print('Image saving completed!')


if __name__ == '__main__':
    test()