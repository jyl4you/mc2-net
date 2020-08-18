import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.config.experimental_run_functions_eagerly(True)
import time
from tqdm import tqdm

from model import MC_Net, vgg_layers, make_custom_loss
from dataset import datalist_loader, train_batch_data_loader, batch_data_loader
from utils import rot_tra_argumentation

tf.random.set_seed(22)
np.random.seed(22)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1,
                     'Batch size (Default: 1)')
flags.DEFINE_integer('image_size', 256,
                     'Image size (size x size) (Default: 256)')
flags.DEFINE_integer('iter_interval', 1,
                     'Iteration interval for logging (Default: 1)')
flags.DEFINE_float('lambda_ssim', 1,
                   'Weight for SSIM loss (Default: 1)')
flags.DEFINE_float('lambda_vgg', 1e-2,
                   'Weight for VGG loss (Default: 0.01)')
flags.DEFINE_float('learning_rate', 1e-4,
                   'Initial learning rate for Adam (Default: 0.0001)')
flags.DEFINE_string('load_weight_name', None,
                    'Load weight of given name (Default: None)')
flags.DEFINE_integer('num_contrast', 3,
                     'Number of contrasts of MR images (Default: 3)')
flags.DEFINE_integer('num_epoch', 1,
                     'Number of epochs for training (Default: 1)')
flags.DEFINE_integer('num_filter', 16,
                     'Number of filters in the first layer of the encoder (Default: 16)')
flags.DEFINE_integer('num_res_block', 9,
                     'Number of residual blocks (Default: 9)')
flags.DEFINE_string('path_data', './data/',
                    'Data load path (Default: ./data/')
flags.DEFINE_string('path_weight', './weight/',
                    'Weight save path (Default: ./weight/')
flags.DEFINE_string('reg_type', 'NMI',
                    'Registration type of input images (No, Network, or NMI) (Default: NMI)')
flags.DEFINE_integer('save_epoch', 10,
                     'Save weights by every given number of epochs (Default: 10)')


def train(_argv):
    os.makedirs('./logs', exist_ok=True)
    logging.get_absl_handler().use_absl_log_file('log', "./logs")
    logging.get_absl_handler().setFormatter(None)

    os.makedirs(FLAGS.path_weight, exist_ok=True)

    model = MC_Net(img_size=FLAGS.image_size,
                   num_filter=FLAGS.num_filter,
                   num_contrast=FLAGS.num_contrast,
                   num_res_block=FLAGS.num_res_block)

    loss_model = vgg_layers(['block3_conv1'])
    final_loss = make_custom_loss(FLAGS.lambda_ssim, FLAGS.lambda_vgg, loss_model)
    model.compile(optimizer=keras.optimizers.Adam(FLAGS.learning_rate),
                  loss=final_loss)
    input_shape = [(None, FLAGS.image_size, FLAGS.image_size, 1)]
    model.build(input_shape=input_shape * FLAGS.num_contrast)
    model.summary()

    # Data loading assumes that the number of contrasts is 3 and contrasts are T1, T2, and FL.
    # If you have different datasets, please modify dataset.datalist_loader.
    y_train_datalist, x_train_datalist = datalist_loader(FLAGS.path_data, FLAGS.reg_type, 'train')
    y_valid_datalist, x_valid_datalist = datalist_loader(FLAGS.path_data, FLAGS.reg_type, 'valid')

    batch_size = FLAGS.batch_size
    epochs = FLAGS.num_epoch
    train_size = len(y_train_datalist[0])
    batch_number = int(np.ceil(train_size//batch_size))

    min_val_loss = 100000

    if FLAGS.load_weight_name is not None:
        weight_path = FLAGS.path_weight + '/' + FLAGS.load_weight_name
        model.load_weights(weight_path)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = [0, 0, 0]
        y_train_datalist_shuffle, x_train_datalist_shuffle =\
            train_batch_data_loader(y_train_datalist+x_train_datalist, FLAGS.num_contrast)
        for batch_index in tqdm(range(batch_number), ncols=100):
            start = batch_index*batch_size

            y_train_datalist_batch = []
            x_train_datalist_batch = []
            for i in range(FLAGS.num_contrast):
                y_train_datalist_batch.append(y_train_datalist_shuffle[i][start:start+batch_size])
                x_train_datalist_batch.append(x_train_datalist_shuffle[i][start:start+batch_size])

            y_train_batch = batch_data_loader(y_train_datalist_batch, FLAGS.num_contrast)
            x_train_batch = batch_data_loader(x_train_datalist_batch, FLAGS.num_contrast)
            y_train_batch, x_train_batch = rot_tra_argumentation(y_train_batch, x_train_batch, FLAGS.num_contrast)

            batch_size_tmp = x_train_batch[0].shape[0]
            tmp_loss = model.train_on_batch(x_train_batch, y_train_batch)

            if batch_index % FLAGS.iter_interval == 0:
                logging.info(f'Epoch [{epoch+1:4d}/{epochs:4d}] | Iter [{batch_index:4d}/{batch_number:4d}] '
                             f'{time.time() - start_time:.2f}s.. '
                             f'train loss for T1: {tmp_loss[0]:.4f}, T2: {tmp_loss[1]:.4f}, FL: {tmp_loss[2]:.4f}')

            train_loss = [(x + y*batch_size_tmp) for (x, y) in zip(train_loss, tmp_loss)]

        train_loss = [x / train_size for x in train_loss]

        print(f'Epoch [{epoch+1:4d}/{epochs:4d}] {time.time() - start_time:.2f}s.. '
              f'train loss for T1: {train_loss[0]:.4f}, T2: {train_loss[1]:.4f}, FL: {train_loss[2]:.4f}')
        logging.info(f'Epoch [{epoch+1:4d}/{epochs:4d}] {time.time() - start_time:.2f}s.. '
                     f'train loss for T1: {train_loss[0]:.4f}, T2: {train_loss[1]:.4f}, FL: {train_loss[2]:.4f}')

        if ((epoch+1) % FLAGS.save_epoch) == 0:
            x_valid = batch_data_loader(x_valid_datalist, FLAGS.num_contrast)
            y_valid = batch_data_loader(y_valid_datalist, FLAGS.num_contrast)
            val_loss = model.evaluate(x_valid, y_valid, verbose=0)
            model.save_weights(f'{FLAGS.path_weight}weight_e{epoch+1:04d}.h5', overwrite=True)

            if np.mean(val_loss) < min_val_loss:
                model.save_weights(f'{FLAGS.path_weight}weight_min_val_loss.h5', overwrite=True)
                min_val_loss = np.mean(val_loss)

            print(f'Weight saved! val loss T1: {val_loss[0]:.4f}, T2: {val_loss[1]:.4f}, FL: {val_loss[2]:.4f}')
            del x_valid, y_valid

        if epoch+1 == epochs:
            model.save_weights(f'{FLAGS.path_weight}weight_final.h5',
                               overwrite=True)
            print(f'Weight saved! Training finished.')

if __name__ == '__main__':
    try:
        app.run(train)
    except SystemExit:
        pass