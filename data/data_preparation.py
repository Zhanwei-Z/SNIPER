# import numpy as np
# import datetime
# import tensorflow as tf
# from configs.params import nyc_params, chicago_params
#
# dataset = 'nyc'
# grid = nyc_params.grid
# number_region = nyc_params.number_region
# label = np.load(dataset + '/' + 'label.npy')
# all_data = np.load(dataset + '/' + 'all_data.npy')
#
# all_data_mean_rec = np.zeros(all_data.shape)
# for j in range(number_region):
#     all_data_mean_rec[:20, j, :] = np.repeat(all_data[:20, j, :] \
#                                                  [(np.where(label[:20, j] == 0))].mean(axis=0).reshape(1, -1),
#                                              repeats=20, axis=0)
# nc = 5
# for i in range(20, len(all_data_mean_rec)):
#     for j in range(number_region):
#         if i > 50:
#             all_data_mean_rec[i, j, :] = all_data[i - 50:i, j, :] \
#                                              [(np.where(label[i - 50:i, j] == 0))][-nc:].mean(axis=0)
#         else:
#             all_data_mean_rec[i, j, :] = all_data[:i, j, :] \
#                                              [(np.where(label[:i, j] == 0))][-nc:].mean(axis=0)
# all_data_mean_rec = all_data - all_data_mean_rec
# all_data = np.concatenate([all_data, all_data_mean_rec], axis=2, out=None)
# dict_xy = np.load(dataset + '/' + 'dict_xy.npy', allow_pickle=True).item()
#
#
# def get_angles(pos, i, d_model):
#     angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#     return pos * angle_rates
#
#
# def get_position_embedding(sentence_length, d_model):
#     angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
#                             np.arange(d_model)[np.newaxis, :],
#                             d_model)
#     # print(angle_rads.shape)
#     # sines.shape:[sentence_length,d_model/2]
#     sines = np.sin(angle_rads[:, 0::2])
#     cosines = np.cos(angle_rads[:, 1::2])
#     # [sentence_length,d_model]
#     position_embedding = np.concatenate([sines, cosines], axis=-1)
#     # [1,sentence_length,d_model]
#     return tf.cast(position_embedding, dtype=tf.float32)
#
#
# position_embedding_x = get_position_embedding(grid, all_data.shape[-1] / 2)
# position_embedding_y = get_position_embedding(grid, all_data.shape[-1] / 2)
# position_embedding_xy = np.zeros((number_region, all_data.shape[-1]))
# for i in range(number_region):
#     position_embedding_xy[i, :] = np.concatenate(
#         [position_embedding_x[dict_xy[i][0]], position_embedding_y[dict_xy[i][1]]], axis=-1)
# position_embedding_xy = position_embedding_xy[np.newaxis, ...]
# position_embedding_xy = np.repeat(position_embedding_xy, repeats=all_data.shape[0], axis=0)
# position_embedding_xy = tf.cast(position_embedding_xy, dtype=tf.float32)
# print(position_embedding_xy.shape)
# temporal_embedding_zt = tf.repeat(tf.reshape(get_position_embedding(all_data.shape[0], all_data.shape[-1]),
#                                              shape=(all_data.shape[0], 1, all_data.shape[-1])), repeats=number_region,
#                                   axis=1)
# print(temporal_embedding_zt.shape)
# all_data = all_data + position_embedding_xy + temporal_embedding_zt
# np.save(file=dataset + '/' + 'data_nyc.npy', arr=all_data)
