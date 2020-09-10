import os
import sys

import tensorflow as tf
import scipy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tf_util


def placeholder_inputs(batch_size, img_rows=66, img_cols=200, separately=False):
    imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, 3))
    if separately:
        speeds_pl = tf.placeholder(tf.float32, shape=(batch_size))
        angles_pl = tf.placeholder(tf.float32, shape=(batch_size))
        labels_pl = [speeds_pl, angles_pl]
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size))
    return imgs_pl, labels_pl


def get_model(net, is_training, bn_decay=None, separately=False):
    """ NVIDIA regression model, input is BxWxHx3, output Bx2"""
    batch_size = net.get_shape()[0].value

    net = tf_util.conv2d(net, 24, [5, 5], padding='VALID', stride=[2, 2], bn=True, is_training=is_training,
                             scope="conv1", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 36, [5, 5], padding='VALID', stride=[2, 2], bn=True, is_training=is_training,
                             scope="conv2", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 48, [5, 5], padding='VALID', stride=[2, 2], bn=True, is_training=is_training,
                             scope="conv3", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [3, 3], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope="conv4", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [3, 3], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope="conv5", bn_decay=bn_decay)

    net = tf.reshape(net, [batch_size, -1])
    net = tf.expand_dims(net, 0)
    
    lstms = []
    for hidden in [256]:
        lstms.append(tf.nn.rnn_cell.BasicLSTMCell(hidden, state_is_tuple=True))
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstms, state_is_tuple=True)
    
    net_unpack = tf.unstack(net, axis=1)
    print(net_unpack[0].shape)
    
    begin_state = stacked_lstm.zero_state(1, dtype=tf.float32)
    
    output, state = tf.nn.static_rnn(stacked_lstm,
                              net_unpack,
                              dtype=tf.float32,
                              initial_state=begin_state)
    
    hidden_out = tf.stack(output, axis=1, name='pack_rnn_outputs')
    # print(hidden_out.shape,'hidden_out')
    hidden_out = tf.reshape(hidden_out, [batch_size, -1])
    
    net = tf_util.fully_connected(hidden_out, 1, activation_fn=None, scope='fc5')

    return net

def cnn_lstm_block(input_tensor):
    lstm_in = tf.reshape(input_tensor, [-1, 28, 28])
    lstm_out = tf_util.stacked_lstm(lstm_in,
                                    num_outputs=10,
                                    time_steps=28,
                                    scope="cnn_lstm")

    W_final = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
    b_final = tf.Variable(tf.truncated_normal([1], stddev=0.1))
    return tf.multiply(tf.atan(tf.matmul(lstm_out, W_final) + b_final), 2)


def get_loss(pred, label, l2_weight=0.0001):
    diff = tf.square(tf.subtract(pred, label))
    loss = (1+tf.abs(label)/180.0*3.1415926)*diff
    
    train_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars[1:]]) * l2_weight

    loss = tf.reduce_mean(loss+l2_loss)
    tf.summary.scalar('loss', loss)

    return loss


def summary_scalar(pred, label):
    threholds = [5, 4, 3, 2, 1, 0.5]
    # angles = [float(t) / 180  for t in threholds]
    angles = [float(t)  for t in threholds]
    # speeds = [float(t)  for t in threholds]

    for i in range(len(threholds)):
        scalar_angle = "angle(" + str(angles[i]) + ")"
        # scalar_speed = "speed(" + str(speeds[i]) + ")"
        # ac_angle = tf.abs(tf.subtract(pred, label)) < float(threholds[i])* scipy.pi / 180
        ac_angle = tf.abs(tf.subtract(pred, label)) < float(threholds[i])* scipy.pi / 180
        # ac_speed = tf.abs(tf.subtract(pred[:, 0], label[:, 0])) < float(threholds[i])/55
        ac_angle = tf.reduce_mean(tf.cast(ac_angle, tf.float32))
        # ac_speed = tf.reduce_mean(tf.cast(ac_speed, tf.float32))

        tf.summary.scalar(scalar_angle, ac_angle)
        # tf.summary.scalar(scalar_speed, ac_speed)


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 66, 200, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
