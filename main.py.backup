# ==============================================================================
# Copyright David Macedo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
import time
from elm import GenELMClassifier
from random_layer import MLPRandomLayer
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
mpl.use('pdf')

NUMBER_OF_FEATURES = 128
BATCH_SIZE = 55
BATCHES_IN_EPOCH = 1000
TRAIN_SIZE = BATCHES_IN_EPOCH * BATCH_SIZE
TEST_SIZE = 10000
NUMBER_OF_EPOCHS = 3
NUMBER_OF_EXPERIMENTS = 2

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
converter = np.array([0,1,2,3,4,5,6,7,8,9])

svm_accuracy = {"LK-SVM":0, "GK-SVM":0}
experiment_accuracy = {"1024HL-ELM":0, "4096HL-ELM":0, "ConvNet":0, "ConvNetSVM":0}

train_features = np.zeros((TRAIN_SIZE, 28 * 28), dtype=float)
train_labels = np.zeros(TRAIN_SIZE, dtype=int)
test_features = mnist.test.images
test_labels = np.zeros(TEST_SIZE, dtype=int)

train_features_cnn = np.zeros((TRAIN_SIZE, NUMBER_OF_FEATURES), dtype=float)
train_labels_cnn = np.zeros(TRAIN_SIZE, dtype=int)
test_labels_cnn = np.zeros(TEST_SIZE, dtype=int)

def print_screen(ndarrayinput, stringinput):
    print("\n"+stringinput)
    print(ndarrayinput.shape)
    print(type(ndarrayinput))
    print(np.mean(ndarrayinput))
    print(ndarrayinput)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def SVM(krnl):
    print("\n##################################\n", krnl, "Kernel SVM Train/Test\n##################################")

    for i in range(BATCHES_IN_EPOCH):
        train_batch = mnist.train.next_batch(BATCH_SIZE)
        features_batch = train_batch[0]
        labels_batch = train_batch[1]
        for j in range(BATCH_SIZE):
            for k in range(28*28):
                train_features[BATCH_SIZE * i + j, k] = features_batch[j, k]
            train_labels[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

    print_screen(train_features, "train_features")
    print_screen(train_labels, "train_labels")

    for j in range(TEST_SIZE):
        test_labels[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

    print_screen(test_features, "test_features")
    print_screen(test_labels, "test_labels")

    initial_time = time.time()

    clf = svm.SVC(kernel=krnl)
    clf.fit(train_features, train_labels)
    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    accuracy = clf.score(test_features, test_labels)
    test_time = time.time() - (training_time + initial_time)
    print("\nTest Time = ", test_time)

    print("\n", krnl, "kernel SVM accuracy =", accuracy)
    return accuracy

def ELM(nodes):
    print("\n############################\n", nodes, "Hidden Layer Nodes ELM Train/Test\n############################")

    for i in range(BATCHES_IN_EPOCH):
        train_batch = mnist.train.next_batch(BATCH_SIZE)
        features_batch = train_batch[0]
        labels_batch = train_batch[1]
        for j in range(BATCH_SIZE):
            for k in range(28*28):
                train_features[BATCH_SIZE * i + j, k] = features_batch[j, k]
            train_labels[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

    print_screen(train_features, "train_features")
    print_screen(train_labels, "train_labels")

    for j in range(TEST_SIZE):
        test_labels[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

    print_screen(test_features, "test_features")
    print_screen(test_labels, "test_labels")

    initial_time = time.time()

    srhl_tanh = MLPRandomLayer(n_hidden=nodes, activation_func="tanh")
    clf = GenELMClassifier(hidden_layer=srhl_tanh)
    clf.fit(train_features, train_labels)
    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    accuracy = clf.score(test_features, test_labels)
    test_time = time.time() - (training_time + initial_time)
    print("\nTest Time = ", test_time)

    print("\n", nodes, "hidden layer nodes ELM accuracy =", accuracy)
    return accuracy

def ConvNet(number_of_training_epochs):
    print("\n############################\nConvNet Train/Test\n############################\n")
    initial_time = time.time()

    for i in range(number_of_training_epochs * BATCHES_IN_EPOCH):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i%BATCHES_IN_EPOCH == 0:
            train_accuracy = model_accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("epoch ", int(i/BATCHES_IN_EPOCH), "training accuracy ", train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    accuracy = model_accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    test_time = time.time() - (training_time + initial_time)
    print("\nTest Time = ", test_time)

    print("\nConvNet accuracy =", accuracy)
    return accuracy

def ConvNetSVM():
    print("\n############################\nConvNetSVM Train/Test\n############################")
    initial_time = time.time()

    for i in range(BATCHES_IN_EPOCH):
        train_batch = mnist.train.next_batch(BATCH_SIZE)
        features_batch = h_fc1.eval(feed_dict={x: train_batch[0]})
        labels_batch = train_batch[1]
        for j in range(BATCH_SIZE):
            for k in range(NUMBER_OF_FEATURES):
                train_features_cnn[BATCH_SIZE * i + j, k] = features_batch[j, k]
            train_labels_cnn[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

    print_screen(train_features_cnn, "train_features_cnn")
    print_screen(train_labels_cnn, "train_labels_cnn")

    test_features_cnn = h_fc1.eval(feed_dict={x: mnist.test.images})
    for j in range(TEST_SIZE):
        test_labels_cnn[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

    print_screen(test_features_cnn, "test_features_cnn")
    print_screen(test_labels_cnn, "train_labels_cnn")

    clf = svm.SVC()
    clf.fit(train_features_cnn, train_labels_cnn)
    training_time = time.time()-initial_time
    print("\nTraining Time = ", training_time)

    accuracy = clf.score(test_features_cnn, test_labels_cnn)
    test_time = time.time() - (training_time + initial_time)
    print("\nTest Time = ", test_time)

    print("\nConvNetSVM accuracy =", accuracy)
    return accuracy

print("\n############################\nStarting\n############################\n")

sess = tf.InteractiveSession()

print("\n############################\nBuilding ConvNet\n############################")

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7 * 7 * 64, NUMBER_OF_FEATURES])
b_fc1 = bias_variable([NUMBER_OF_FEATURES])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([NUMBER_OF_FEATURES, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

print("\n############################\nExecuting Experiments\n############################")

df_svm = pd.DataFrame()
df_global = pd.DataFrame()

#svm_accuracy["LK-SVM"] = SVM("linear")
#svm_accuracy["GK-SVM"] = SVM("rbf")

df_svm = df_svm.append(svm_accuracy, ignore_index=True)
df_svm = df_svm[["LK-SVM", "GK-SVM"]]

for index in range(NUMBER_OF_EXPERIMENTS):
    experiment_accuracy["1024HL-ELM"] = ELM(1024)
    experiment_accuracy["4096HL-ELM"] = ELM(4096)
    experiment_accuracy["ConvNet"] = ConvNet(NUMBER_OF_EPOCHS)
    experiment_accuracy["ConvNetSVM"] = ConvNetSVM()
    df_global = df_global.append(experiment_accuracy, ignore_index=True)

df_global = df_global[["1024HL-ELM", "4096HL-ELM", "ConvNet", "ConvNetSVM"]]

print("\n############################\nPrinting Results\n############################\n")

print("\n", df_svm)
print("\n", df_global,"\n")
print(df_global.describe())
print("\n", df_global.describe().transpose())

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
#plt.tight_layout()

width = 3.487
height = width / 1.618

fig=plt.figure(figsize=(width,height))
#fig=plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

#ax = df_global.plot.box()
ax = df_global.plot.box(figsize=(width,height))
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_title("Title")

fig.set_size_inches(width, height)
plt.savefig("df_global.pdf")

print("\n############################\nStoping\n############################\n")

sess.close()
