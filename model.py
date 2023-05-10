import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, losses, backend

# setting
time_len = 1

# model setting
is_space_attention = True
is_temporal_attention = True
lr = 1e-3
epochs = 300
sample_len, channels_num = int(128 * time_len), 64
cnn_kernel_num = 5
cnn_block_len = 4

# space attention setting
sa_kq = 50
sa_block_num = math.ceil(time_len)
sa_channel_dense_num = cnn_kernel_num * sa_block_num

# temporal attention setting
ta_do_percent = 0.5
ta_block_len = channels_num


class MySpaceAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MySpaceAttention, self).__init__(**kwargs)
        se_cnn_num = 10
        self.my_se_dense = keras.models.Sequential([
            keras.layers.Conv2D(se_cnn_num, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='elu'),
            keras.layers.Permute((1, 3, 2)),
            keras.layers.MaxPool2D((1, se_cnn_num)),
            keras.layers.Dropout(ta_do_percent),
            keras.layers.Dense(8, activation='elu'),
            keras.layers.Dropout(ta_do_percent),
            keras.layers.Dense(channels_num, activation='elu'),
            # keras.layers.Softmax(-1),
        ])

    def build(self, input_shape):
        super(MySpaceAttention, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.reshape(x, (-1, int(sample_len / sa_block_num), sa_block_num, channels_num))
        x = tf.transpose(x, [0, 1, 3, 2])
        w = self.my_se_dense(x)
        x = tf.transpose(x, [0, 1, 3, 2])

        # SE attention
        y = w * x
        y = tf.reshape(y, (-1, sample_len, channels_num))

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class MyTemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyTemporalAttention, self).__init__(**kwargs)
        self.dense_k = keras.models.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Dense(sa_kq, activation='elu'),
        ])
        self.dense_q = keras.models.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Dense(sa_kq, activation='elu'),
        ])
        self.dense_v = keras.models.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Dense(sa_channel_dense_num, activation='tanh'),
        ])
        self.my_se_softmax = keras.layers.Softmax(1)

    def build(self, input_shape):
        super(MyTemporalAttention, self).build(input_shape) 

    def call(self, x, **kwargs):
        k = self.dense_k(x)
        q = self.dense_q(x)
        v = self.dense_v(x)
        w = backend.batch_dot(k, tf.transpose(q, [0, 2, 1])) / math.sqrt(sample_len)
        w = self.my_se_softmax(w)
        y = backend.batch_dot(w, v)

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


def create_model():
    # set the model
    model = Sequential()
    model.add(keras.layers.BatchNormalization(axis=2, input_shape=(sample_len, channels_num)))

    # Space Attention
    if is_space_attention:
        model.add(MySpaceAttention())

    # cnn
    model.add(keras.layers.Conv1D(cnn_kernel_num, 5, strides=1, padding='same', activation='tanh'))
    model.add(keras.layers.MaxPool1D(cnn_block_len))
    model.add(keras.layers.Reshape((int(sample_len / cnn_block_len / sa_block_num), sa_block_num * cnn_kernel_num)))

    # Temporal Attention
    if is_temporal_attention:
        model.add(MyTemporalAttention())

    # fc
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    # set the optimizers
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )

    return model


def main():
    model = create_model()
    random_data = tf.ones((16, sample_len, channels_num))
    model(random_data)
    model.summary()
    del model

if __name__ == '__main__':
    main()
