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
sys.path.append(os.path.join(BASE_DIR, 'models_dep'))

import provider_steer as provider
# import provider_steer_img5c as provider
import tensorflow as tf
from helper import str2bool


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

FLAGS = parser.parse_args()
BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
ADD_LSTM = FLAGS.add_lstm

supported_models = ["nvidia_io", "nvidia_pn",
                    "resnet152_io", "resnet152_pn",
                    "inception_v4_io", "inception_v4_pn",
                    "densenet169_io", "densenet169_pn", "resnet152_io_speed", "nvidia_io_nvi","vgg16_io"]
# assert (FLAGS.model in supported_models)

# MODEL = importlib.import_module(FLAGS.model)  # import network module
# MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

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


def plot_acc(preds_mae, preds_mse, preds_stlossmae, preds_stlossmse,preds_img5, labels, counts = 100):
    a_list_mae = []
    a_list_mse = []
    a_list_stlossmse = []
    a_list_stlossmae = []
    a_list_img5 = []
    s_list = []
    for i in range(counts):
        # acc_a = np.abs(np.subtract(preds, labels)) < (20.0 / 180 * scipy.pi / counts * i)
        acc_a_mae = np.abs(np.subtract(preds_mae, labels)) < (20.0 / counts * i)
        a_list_mae.append(np.mean(acc_a_mae))
        
        acc_a_mse = np.abs(np.subtract(preds_mse, labels)) < (20.0 / counts * i)
        a_list_mse.append(np.mean(acc_a_mse))
        
        acc_a_stlossmae = np.abs(np.subtract(preds_stlossmae, labels)) < (20.0 / counts * i)
        a_list_stlossmae.append(np.mean(acc_a_stlossmae))
        
        acc_a_stlossmse = np.abs(np.subtract(preds_stlossmse, labels)) < (20.0 / counts * i)
        a_list_stlossmse.append(np.mean(acc_a_stlossmse))
        
        acc_a_img5 = np.abs(np.subtract(preds_img5, labels)) < (20.0 / counts * i)
        a_list_img5.append(np.mean(acc_a_img5))

    # for i in range(counts):
        # acc_s = np.abs(np.subtract(preds[:, 0], labels[:, 0])) < (15.0 / 20 / counts * i)
        # s_list.append(np.mean(acc_s))

    # print (len(a_list), len(s_list))
    a_xaxis = [20.0 / counts * i for i in range(counts)]
    # s_xaxis = [15.0 / counts * i for i in range(counts)]

    # auc_angle = np.trapz(np.array(a_list), x=a_xaxis) / 20.0
    # auc_speed = np.trapz(np.array(s_list), x=s_xaxis) / 15.0

    plt.style.use('ggplot')
    plt.figure()
    # plt.plot(a_xaxis, np.array(a_list_mae), label='Area Under Curve (AUC): %f' % auc_angle)
    plt.plot(a_xaxis, np.array(a_list_mae), label='mae')
    plt.plot(a_xaxis, np.array(a_list_mse), label='mse')
    plt.plot(a_xaxis, np.array(a_list_stlossmae), label='stlossmae')
    plt.plot(a_xaxis, np.array(a_list_stlossmse), label='stlossmse')
    plt.plot(a_xaxis, np.array(a_list_img5), label='img5')
    # plt.legend(loc='best')
    plt.legend(('mae','mse','stlossmae','stlossmse','img5'),loc='upper left')
    plt.xlabel("Threshold (angle)/degree")
    plt.ylabel("Prediction accuracy")
    plt.savefig(os.path.join('result_final', "acc_angle.png"))

def plot_acc_from_txt(counts=30):
    # preds = np.loadtxt(os.path.join(RESULT_DIR, "val/preds_val.txt"))
    preds_mae = np.loadtxt(os.path.join('result_final/pilotnet_io_mae_lstm2', "val/preds_val.txt"))
    preds_mse = np.loadtxt(os.path.join('result_final/pilotnet_io_mse_lstm2', "val/preds_val.txt"))
    preds_stlossmae = np.loadtxt(os.path.join('result_final/pilotnet_io_stlossmae_lstm2', "val/preds_val.txt"))
    preds_stlossmse = np.loadtxt(os.path.join('result_final/pilotnet_io_stlossmse_lstm2', "val/preds_val.txt"))
    
    preds_img5 = np.loadtxt(os.path.join('result_final/pilotnet_io_stlossmae_img5_lstm2', "val/preds_val.txt"))

    # labels = np.loadtxt(os.path.join(RESULT_DIR, "test/labels_val.txt"))
    labels = np.loadtxt(os.path.join('result_final/pilotnet_io_mae_lstm2', "val/labels_val.txt"))
    # print (preds.shape, labels.shape)
    plot_acc(preds_mae, preds_mse, preds_stlossmae, preds_stlossmse,preds_img5, labels, counts)


if __name__ == "__main__":
    # if FLAGS.test: test()
    # else: evaluate()
    # plot_acc_from_txt()
    # evaluate()
    plot_acc_from_txt()
