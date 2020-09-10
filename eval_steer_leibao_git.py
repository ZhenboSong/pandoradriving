import argparse
import importlib
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider_leibao as provider

import tensorflow as tf
from helper import str2bool

sys.path.append(os.path.join(BASE_DIR, 'models_dep'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='nvidia_pn',
                    help='Model name [default: nvidia_pn]')
parser.add_argument('--model_path', default='logs/nvidia_pn/model_best.ckpt',
                    help='Model checkpoint file path [default: logs/nvidia_pn/model_best.ckpt]')
parser.add_argument('--max_epoch', type=int, default=250,
                    help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch Size during training [default: 8]')
parser.add_argument('--result_dir', default='results',
                    help='Result folder path [default: results]')
parser.add_argument('--test', type=str2bool, default=False, # only used in test server
                    help='Get performance on test data [default: False]')
parser.add_argument('--add_lstm', type=bool, default=False,
                    help='Introduce LSTM mechanism in netowrk [default: False]')

parser.add_argument('--add_orientations', type=bool, default=False,
                    help='Add the orientations in netowrk [default: False]')
parser.add_argument('--loss_function', default='mae',
                    help='loss function [default: mae]')

FLAGS = parser.parse_args()
BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
ADD_LSTM = FLAGS.add_lstm
ADD_ORI = FLAGS.add_orientations

supported_loss_func = ["mae", "mse", "stlossmse", "stloss2"]
assert (FLAGS.loss_function in supported_models)
LOSS_FUNC = FLAGS.loss_function

supported_models = ["nvidia_io", "nvidia_pn",
                    "resnet152_io", "resnet152_pn",
                    "inception_v4_io", "inception_v4_pn",
                    "densenet169_io", "densenet169_pn", "resnet152_io_speed", "nvidia_io_nvi","vgg16_io"]
# assert (FLAGS.model in supported_models)

MODEL = importlib.import_module(FLAGS.model)  # import network module
# MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
MODEL_FILE = os.path.join(BASE_DIR, 'models_dep', FLAGS.model+'.py')

RESULT_DIR = os.path.join(FLAGS.result_dir, FLAGS.model)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if FLAGS.test:
    TEST_RESULT_DIR = os.path.join(RESULT_DIR, "test")
    if not os.path.exists(TEST_RESULT_DIR):
        os.makedirs(TEST_RESULT_DIR)
    LOG_FOUT = open(os.path.join(TEST_RESULT_DIR, 'log_test.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')
else:
    VAL_RESULT_DIR = os.path.join(RESULT_DIR, "val")
    if not os.path.exists(VAL_RESULT_DIR):
        os.makedirs(VAL_RESULT_DIR)
    LOG_FOUT = open(os.path.join(VAL_RESULT_DIR, 'log_evaluate.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        if '_pn' in MODEL_FILE:
            data_input = provider.Provider()
            imgs_pl, pts_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
            imgs_pl = [imgs_pl, pts_pl]
        elif '_io' in MODEL_FILE:
            data_input = provider.Provider()
            if ADD_ORI:
                imgs_pl, labels_pl = MODEL.placeholder_inputs_img5(BATCH_SIZE)
            else:
                imgs_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
        else:
            raise NotImplementedError

        is_training_pl = tf.placeholder(tf.bool, shape=())
        print(is_training_pl)

        # Get model and loss
        # pred = MODEL.get_model(imgs_pl, is_training_pl, add_lstm=ADD_LSTM)

        pred = MODEL.get_model(imgs_pl, is_training_pl)

        if LOSS_FUNC == "mae":
            loss = MODEL.get_loss_mae(pred, labels_pl)
        elif LOSS_FUNC == "mse":
            loss = MODEL.get_loss_mse(pred, labels_pl)
        elif LOSS_FUNC == "stlossmse":
            loss = MODEL.get_loss_stlossmse(pred, labels_pl)
        elif LOSS_FUNC == "stloss2":
            loss = MODEL.get_loss_stloss2(pred, labels_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")
    

    ops = {'imgs_pl': imgs_pl,
            'labels_pl': labels_pl,
            # 'speed_pl' : speed_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss}

    eval_one_epoch(sess, ops, data_input)


def eval_one_epoch(sess, ops, data_input):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0
    loss_val_sum = 0

    num_batches = data_input.num_val // BATCH_SIZE
    acc_a_sum = [0] * 5
    acc_s_sum = [0] * 5

    preds = []
    labels_total = []
    acc_a = [0] * 5
    acc_s = [0] * 5
    for batch_idx in range(num_batches):
        if "_io" in MODEL_FILE:

            imgs, labels = data_input.load_one_batch(BATCH_SIZE, "val", reader_type="io", add_ori=ADD_ORI)
            if "resnet" in MODEL_FILE or "inception" in MODEL_FILE or "densenet" in MODEL_FILE or "vgg" in MODEL_FILE:
                imgs = MODEL.resize(imgs)

            feed_dict = {ops['imgs_pl']: imgs,
                         ops['labels_pl']: labels,
                         # ops['speed_pl']: speeds,
                         ops['is_training_pl']: is_training}
        else:
            imgs, others, labels = data_input.load_one_batch(BATCH_SIZE, "val", add_ori=ADD_ORI)
            if "resnet" in MODEL_FILE or "inception" in MODEL_FILE or "densenet" in MODEL_FILE:
                imgs = MODEL.resize(imgs)
            feed_dict = {ops['imgs_pl'][0]: imgs,
                         ops['imgs_pl'][1]: others,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}

        loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                    feed_dict=feed_dict)

        # print(pred_val[:5], np.max(pred_val))
        preds.append(pred_val)
        # print(np.max(pred_val), 'pred_val')
        labels = labels.reshape((16,1))
        labels_total.append(labels)
        # loss_sum += np.mean(np.square(np.subtract(pred_val, labels)))
        loss_sum += np.mean(np.abs(np.subtract(pred_val, labels)))
        loss_val_sum += loss_val
        for i in range(5):
            acc_a[i] = np.mean(np.abs(np.subtract(pred_val, labels)) < (1.0 * (i+1)))
            acc_a_sum[i] += acc_a[i]
            # acc_s[i] = np.mean(np.abs(np.subtract(pred_val[:, 0], labels[:, 0])) < (1.0 * (i+1) / 20))
            # acc_s_sum[i] += acc_s[i]

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('netowrk loss: %f' % (loss_val_sum / float(num_batches)))
    for i in range(5):
        log_string('eval accuracy (angle-%d): %f' % (float(i+1), (acc_a_sum[i] / float(num_batches))))
        # log_string('eval accuracy (speed-%d): %f' % (float(i+1), (acc_s_sum[i] / float(num_batches))))

    preds = np.vstack(preds)
    labels = np.vstack(labels_total)

    a_error = mean_max_error(preds, labels, dicts=get_dicts())
    # log_string('eval error (mean-max): angle:%.2f speed:%.2f' %
               # (a_error / scipy.pi * 180, s_error * 20))
    log_string('eval error (mean-max): angle:%.2f ' %
                (a_error))
    a_error = max_error(preds, labels)
    log_string('eval error (max): angle:%.2f' %
               (a_error))
    a_error = mean_topk_error(preds, labels, 5)
    log_string('eval error (mean-top5): angle:%.2f' %
               (a_error))
    a_error = mean_error(preds, labels)
    log_string('eval error (mean): angle:%.2f' %
               (a_error))
    
    a_sd = mean_sd(preds, labels)
    log_string('eval sd (mean): angle:%.2f' %
               (a_sd))
    
    print (preds.shape, labels.shape)
    np.savetxt(os.path.join(VAL_RESULT_DIR, "preds_val.txt"), preds)
    np.savetxt(os.path.join(VAL_RESULT_DIR, "labels_val.txt"), labels)
    # plot_acc(preds, labels)


def test():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # tf.reset_default_graph()
        if '_pn' in MODEL_FILE:
            data_input = provider.Provider()
            imgs_pl, pts_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
            imgs_pl = [imgs_pl, pts_pl]
        elif '_io' in MODEL_FILE:
            data_input = provider.Provider()
            if ADD_ORI:
                imgs_pl, labels_pl = MODEL.placeholder_inputs_img5(BATCH_SIZE)
            else:
                imgs_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
        else:
            raise NotImplementedError


        is_training_pl = tf.placeholder(tf.bool, shape=())
        print(is_training_pl)

        # Get model and loss
        pred = MODEL.get_model(imgs_pl, is_training_pl, add_lstm=ADD_LSTM)

        if LOSS_FUNC == "mae":
            loss = MODEL.get_loss_mae(pred, labels_pl)
        elif LOSS_FUNC == "mse":
            loss = MODEL.get_loss_mse(pred, labels_pl)
        elif LOSS_FUNC == "stlossmse":
            loss = MODEL.get_loss_stlossmse(pred, labels_pl)
        elif LOSS_FUNC == "stloss2":
            loss = MODEL.get_loss_stloss2(pred, labels_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # sess.run(tf.global_variables_initializer())
    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'imgs_pl': imgs_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss}

    test_one_epoch(sess, ops, data_input)


def test_one_epoch(sess, ops, data_input):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0

    num_batches = data_input.num_test // BATCH_SIZE
    acc_a_sum = [0] * 5
    acc_s_sum = [0] * 5

    preds = []
    labels_total = []
    acc_a = [0] * 5
    acc_s = [0] * 5
    for batch_idx in range(num_batches):
        if "_io" in MODEL_FILE:
            imgs, labels = data_input.load_one_batch(BATCH_SIZE, reader_type="io", add_ori=ADD_ORI)
            if "resnet" in MODEL_FILE or "inception" in MODEL_FILE or "densenet" in MODEL_FILE:
                imgs = MODEL.resize(imgs)
            feed_dict = {ops['imgs_pl']: imgs,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}
        else:
            imgs, others, labels = data_input.load_one_batch(BATCH_SIZE, add_ori=ADD_ORI)
            if "resnet" in MODEL_FILE or "inception" in MODEL_FILE or "densenet" in MODEL_FILE:
                imgs = MODEL.resize(imgs)
            feed_dict = {ops['imgs_pl'][0]: imgs,
                         ops['imgs_pl'][1]: others,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}

        loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                    feed_dict=feed_dict)

        preds.append(pred_val)
        labels_total.append(labels)
        loss_sum += np.mean(np.square(np.subtract(pred_val, labels)))
        for i in range(5):
            acc_a[i] = np.mean(np.abs(np.subtract(pred_val[:, 1], labels[:, 1])) < (1.0 * (i+1) / 180 * scipy.pi))
            acc_a_sum[i] += acc_a[i]
            acc_s[i] = np.mean(np.abs(np.subtract(pred_val[:, 0], labels[:, 0])) < (1.0 * (i+1) / 20))
            acc_s_sum[i] += acc_s[i]

    log_string('test mean loss: %f' % (loss_sum / float(num_batches)))
    for i in range(5):
        log_string('test accuracy (angle-%d): %f' % (float(i+1), (acc_a_sum[i] / float(num_batches))))
        log_string('test accuracy (speed-%d): %f' % (float(i+1), (acc_s_sum[i] / float(num_batches))))

    preds = np.vstack(preds)
    labels = np.vstack(labels_total)

    a_error, s_error = mean_max_error(preds, labels, dicts=get_dicts(description="test"))
    log_string('test error (mean-max): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = max_error(preds, labels)
    log_string('test error (max): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = mean_topk_error(preds, labels, 5)
    log_string('test error (mean-top5): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = mean_error(preds, labels)
    log_string('test error (mean): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))

    print (preds.shape, labels.shape)
    np.savetxt(os.path.join(TEST_RESULT_DIR, "preds_val.txt"), preds)
    np.savetxt(os.path.join(TEST_RESULT_DIR, "labels_val.txt"), labels)
    # plot_acc(preds, labels)


def plot_acc(preds, labels, counts = 100):
    a_list = []
    s_list = []
    for i in range(counts):
        # acc_a = np.abs(np.subtract(preds, labels)) < (20.0 / 180 * scipy.pi / counts * i)
        acc_a = np.abs(np.subtract(preds, labels)) < (20.0 / counts * i)
        a_list.append(np.mean(acc_a))

    # for i in range(counts):
        # acc_s = np.abs(np.subtract(preds[:, 0], labels[:, 0])) < (15.0 / 20 / counts * i)
        # s_list.append(np.mean(acc_s))

    # print (len(a_list), len(s_list))
    a_xaxis = [20.0 / counts * i for i in range(counts)]
    # s_xaxis = [15.0 / counts * i for i in range(counts)]

    auc_angle = np.trapz(np.array(a_list), x=a_xaxis) / 20.0
    # auc_speed = np.trapz(np.array(s_list), x=s_xaxis) / 15.0

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(a_xaxis, np.array(a_list), label='Area Under Curve (AUC): %f' % auc_angle)
    plt.legend(loc='best')
    plt.xlabel("Threshold (angle)/degree")
    plt.ylabel("Prediction accuracy")
    plt.savefig(os.path.join(RESULT_DIR, "acc_angle.png"))

def plot_acc_from_txt(counts=100):
    preds = np.loadtxt(os.path.join(RESULT_DIR, "val/preds_val.txt"))
    labels = np.loadtxt(os.path.join(RESULT_DIR, "val/labels_val.txt"))
    print (preds.shape, labels.shape)
    plot_acc(preds, labels, counts)

def get_dicts(description="val"):
    if description == "train":
        raise NotImplementedError
    elif description == "val": # batch_size == 8
        # return [120] * 4 + [111] + [120] * 4 + [109] + [120] * 9 + [89 - 87 % 8]
        return [1472] + [1654] + [1164]
    elif description == "test": # batch_size == 8
        # return [120] * 9 + [116] + [120] * 4 + [106] + [120] * 4 + [114 - 114 % 8]
        return [109] + [120] * 6 + [107] + [120] * 4 + [112]
    else:
        raise NotImplementedError

def mean_max_error(preds, labels, dicts):
    cnt = 0
    a_error = 0
    s_error = 0
    for i in dicts:
        a_error += np.max(np.abs(preds[cnt:cnt+i] - labels[cnt:cnt+i]))
        print(np.max(np.abs(preds[cnt:cnt+i] - labels[cnt:cnt+i])), 'a_error')
        # s_error += np.max(np.abs(preds[cnt:cnt+i, 0] - labels[cnt:cnt+i, 0]))
        cnt += i
    return a_error / float(len(dicts))

def max_error(preds, labels):
    return np.max(np.abs(preds - labels))

def mean_error(preds, labels):
    return np.mean(np.abs(preds - labels))

def mean_topk_error(preds, labels, k):
    a_error = np.abs(preds - labels)
    # s_error = np.abs(preds[:,0] - labels[:,0])
    return np.mean(np.sort(a_error)[::-1][0:k])

def mean_sd(preds, labels):
    print(preds.shape,'pred')
    # preds = np.array(list(preds)).reshape((1174,))
    a_sd = np.std(preds,axis=0)
    return a_sd

if __name__ == "__main__":
    if FLAGS.test: test()
    else: evaluate()
    plot_acc_from_txt()
