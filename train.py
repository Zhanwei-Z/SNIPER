import numpy as np
import tensorflow as tf
import math
import gc
import time
import os
from model import SNIPER
import argparse
from lib.utils import get_neigh_index, prepare_data, loss_function, compute_loss, get_f1_threshold, get_metrics, \
    EarlyStopping
from configs.params import nyc_params, chicago_params
tf.random.set_seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, help="test program")
parser.add_argument("--dataset", type=str, help="test program")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset = args.dataset
if dataset == 'nyc':
    params = nyc_params
    print('here')
elif dataset == 'chicago':
    params = chicago_params
else:
    raise NameError

len_recent_time = params.len_recent_time
number_region = params.number_region
threshold_nc = dataset + '/' + params.threshold_nc
label = dataset + '/' + params.label
all_data = dataset + '/' + params.all_data
dict_xy = np.load(dataset + '/' + params.dict_xy, allow_pickle=True).item()
threshold_nc = np.load(file=threshold_nc)
label = np.load(file=label)
label = tf.cast(label, dtype=tf.float32)
all_data = np.load(file=all_data)
neigh_road_index = get_neigh_index(dataset + '/' + 'road_ad.txt')
neigh_record_index = get_neigh_index(dataset + '/' + 'record_ad.txt')
neigh_poi_index = get_neigh_index(dataset + '/' + 'poi_ad.txt')
all_data = prepare_data(all_data, len_recent_time)
threshold_nc = prepare_data(threshold_nc, len_recent_time)
label = label[len_recent_time:]
print(all_data.shape, threshold_nc.shape, label.shape)

train_x = all_data[:int(len(all_data) * 0.6)]
train_y = label[:int(len(label) * 0.6)]
train_threshold_nc = threshold_nc[:int(len(threshold_nc) * 0.6)]
val_x = all_data[int(len(all_data) * 0.6):int(len(all_data) * 0.8)]
val_y = label[int(len(label) * 0.6):int(len(label) * 0.8)]
val_threshold_nc = threshold_nc[int(len(threshold_nc) * 0.6):int(len(threshold_nc) * 0.8)]
test_x = all_data[int(len(all_data) * 0.8):]
test_y = label[int(len(label) * 0.8):]
test_threshold_nc = threshold_nc[int(len(threshold_nc) * 0.8):]
gc.collect()

learning_rate = params.learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
dr = params.dr
number_sp = params.number_sp
model = SNIPER(dr, len_recent_time, number_sp, number_region, neigh_poi_index, neigh_road_index, neigh_record_index)


@tf.function
def train_one_step(x, label_y):
    with tf.GradientTape() as tape:
        all_data_static, threshold_nc1, all_data_dynamic_now = x
        y_predict, y_dy, dy_diff = model(all_data_static, threshold_nc1, all_data_dynamic_now)
        loss, focal_loss, dy_loss = loss_function(y_predict, label_y, dy_diff)
        loss = tf.reduce_mean(loss)
        tf.print('training:', "loss:", loss, "   focal loss:", tf.reduce_mean(focal_loss))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return y_dy


patience = params.patience
delta = params.delta
early_stop = EarlyStopping(patience=patience, delta=delta)

batch_size = params.batch_size
batch_train = math.ceil((len(train_x)) / batch_size)
batch_val = math.ceil((len(val_x)) / batch_size)
training_epoch = params.training_epoch
start = time.time()
for epoch in range(0, training_epoch):
    i = 0
    y_dynamic = tf.ones((len_recent_time, number_region, 2 * dr))
    train_x_batch = [train_x[i * batch_size:(i + 1) * batch_size],
                     train_threshold_nc[i * batch_size:(i + 1) * batch_size],
                     y_dynamic]
    train_y_batch = train_y[i * batch_size:(i + 1) * batch_size]
    y_dynamic = train_one_step(train_x_batch, train_y_batch)

    for i in range(1, batch_train):
        print('epoch:', epoch, 'i:', i)
        train_x_batch = [train_x[i * batch_size:(i + 1) * batch_size],
                         train_threshold_nc[i * batch_size:(i + 1) * batch_size], y_dynamic]
        train_y_batch = train_y[i * batch_size:(i + 1) * batch_size]
        y_dynamic = train_one_step(train_x_batch, train_y_batch)

    val_loss = compute_loss(val_x, val_threshold_nc, y_dynamic, val_y, model, batch_size)
    print('val_loss:', val_loss)
    early_stop(val_loss)
    if early_stop.early_stop:
        break
    else:
        print('update')
end = time.time()
print(end - start)

threshold_f1, threshold_accu, y_dy_valid = \
    get_f1_threshold(val_x, val_threshold_nc, y_dynamic, val_y, model, batch_size)
ap_score, ra_score, f1, recall, precision, accu, y, test_predict = \
    get_metrics(test_x, test_threshold_nc, y_dy_valid, test_y, model, batch_size, threshold_f1, threshold_accu)
