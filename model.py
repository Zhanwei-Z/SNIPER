import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
import math

class Evolution(Layer):

    def __init__(self, dr2, **kwargs):
        self.dr2 = dr2
        super(Evolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name='wl',
                                  shape=(2 * self.dr2, self.dr2),
                                  initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.5, seed=2021),
                                  trainable=True)

        super(Evolution, self).build(input_shape)  # Be sure to call this at the end

    def call(self, all_data_static, threshold_nc, all_data_dynamic_now):
        all_data_dynamic = tf.expand_dims(all_data_dynamic_now, 0)
        all_data_dynamic_now=tf.sigmoid(tf.matmul(tf.concat([all_data_dynamic_now,all_data_static[0]],axis=-1),self.w1)\
                                        *tf.repeat(thre_nc[0],self.dr2,axis=-1)+all_data_dynamic_now\
                                        *tf.repeat(1-thre_nc[0],self.dr2,axis=-1)) * math.exp(-1/2)
        all_data_dynamic_diff = []
        for i in range(1, len(threshold_nc)):
            all_data_dynamic_now_diff = all_data_dynamic_now
            all_data_dynamic_now = tf.sigmoid(
                tf.matmul(tf.concat([all_data_dynamic_now, all_data_static[i]], axis=-1), self.w1)
                * tf.repeat(threshold_nc[i], self.dr2, axis=-1) + all_data_dynamic_now
                * tf.repeat(1 - threshold_nc[i], self.dr2, axis=-1)) * math.exp(-1/2)
            all_data_dynamic_now_diff = all_data_dynamic_now - all_data_dynamic_now_diff
            all_data_dynamic_diff.append(tf.expand_dims(all_data_dynamic_now_diff, 0))
            all_data_dynamic = tf.concat([all_data_dynamic, tf.expand_dims(all_data_dynamic_now, 0)], axis=0)

        all_data_dynamic_diff = tf.concat(all_data_dynamic_diff, axis=0)
        return all_data_dynamic, all_data_dynamic_now, all_data_dynamic_diff


class Attention(Layer):

    def __init__(self, dr2, len_recent_time, number_region, **kwargs):
        self.dr2 = dr2
        self.len_recent_time = len_recent_time
        self.number_region = number_region
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wq = self.add_weight(
            shape=(self.dr2, self.dr2),
            initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=2021),
            trainable=True)
        self.wk = self.add_weight(
            shape=(self.dr2, self.dr2),
            initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=2021),
            trainable=True)
        self.wd_s = self.add_weight(
            shape=(self.dr2, self.dr2),
            initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=2021),
            trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, data, neigh_index):  # len,time,regions,features
        data_neigh = tf.nn.embedding_lookup(tf.transpose(data, (2, 0, 1, 3)),
                                            neigh_index)  # regions,len,time,features->regions,neigh,len,time,features
        data_neigh = tf.transpose(data_neigh, (2, 3, 0, 1, 4))  # len,time,regions,neigh,features
        data = tf.reshape(data, (-1, data.shape[1], data.shape[2], 1, data.shape[-1]))
        data = tf.matmul(data, self.wq)
        data_neigh = tf.matmul(data_neigh, self.wk)
        out = tf.matmul(tf.nn.softmax(tf.matmul(data, data_neigh, transpose_b=True), axis=-1), data_neigh)
        out = data + out
        out = tf.sigmoid(
            tf.matmul(tf.reshape(out, (-1, self.len_recent_time, self.number_region, self.dr2)), self.wd_s))
        return out


class MultiAttention(Layer):

    def __init__(self, num_sp, dr2, len_recent_time, number_region, **kwargs):
        self.dr2 = dr2
        self.num_sp = num_sp
        self.attention_layers_poi = [Attention(self.dr2, len_recent_time, number_region) for _ in range(self.num_sp)]
        self.attention_layers_road = [Attention(self.dr2, len_recent_time, number_region) for _ in range(self.num_sp)]
        self.attention_layers_record = [Attention(self.dr2, len_recent_time, number_region) for _ in range(self.num_sp)]

        super(MultiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_poi = self.add_weight(
            shape=(self.dr2,),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        self.w_road = self.add_weight(
            shape=(self.dr2,),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        self.w_record = self.add_weight(
            shape=(self.dr2,),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        super(MultiAttention, self).build(input_shape)

    def call(self, all_data, neigh_poi_index, neigh_road_index, neigh_record_index):  #
        all_data_static_poi = all_data
        all_data_static_road = all_data
        all_data_static_record = all_data
        for i in range(self.num_sp):
            all_data_static_poi = self.attention_layers_poi[i](all_data_static_poi, neigh_poi_index)
            all_data_static_road = self.attention_layers_road[i](all_data_static_road, neigh_road_index)
            all_data_static_record = self.attention_layers_record[i](all_data_static_record, neigh_record_index)
        out = tf.sigmoid(all_data_static_poi * self.w_poi + all_data_static_road * self.w_road +
                         all_data_static_record * self.w_record)
        return out


class SNIPER(tf.keras.models.Model):
    def __init__(self, dr, len_recent_time, number_sp, number_region, neigh_poi_index, neigh_road_index,
                 neigh_record_index, **kwargs):
        super(SNIPER, self).__init__(**kwargs)
        self.neigh_poi_index = neigh_poi_index
        self.neigh_road_index = neigh_road_index
        self.neigh_record_index = neigh_record_index
        self.evolution = Evolution(dr * 2)
        self.multiattention = [MultiAttention(number_sp, 2 * dr, len_recent_time, number_region) for _ in range(2)]
        self.convlstm = keras.layers.ConvLSTM2D(1, 1, strides=(1, 1),
                                                padding='valid',
                                                data_format=None,
                                                dilation_rate=(1, 1),
                                                activation='tanh',
                                                recurrent_activation='hard_sigmoid',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                recurrent_initializer='orthogonal',
                                                bias_initializer='zeros',
                                                unit_forget_bias=True,
                                                return_sequences=False,
                                                )
        self.final_layer = keras.layers.Dense(number_region, activation='sigmoid', bias_initializer='ones')

    def call(self, all_data_static, threshold_nc, all_data_dynamic_now):
        all_data_dynamic, all_data_dynamic_now, all_data_dynamic_diff = self.evolution(all_data_static, threshold_nc,
                                                                                       all_data_dynamic_now)

        all_data_dynamic = self.multiattention[0](all_data_dynamic, self.neigh_poi_index, self.neigh_road_index,
                                                  self.neigh_record_index)
        all_data_static = self.multiattention[1](all_data_static, self.neigh_poi_index, self.neigh_road_index,
                                                 self.neigh_record_index)
        all_data_dynamic = tf.expand_dims(all_data_dynamic, 3)
        all_data_static = tf.expand_dims(all_data_static, 3)
        all_data = tf.concat([all_data_dynamic, all_data_static], axis=-1)
        all_data = self.convlstm(all_data)
        all_data = tf.reshape(all_data, (-1, all_data.shape[1]))
        all_data = self.final_layer(all_data)
        # print(all_data.shape)
        return all_data, all_data_dynamic_now, all_data_dynamic_diff
