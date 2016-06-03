from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
import tensorflow as tf
import numpy as np

number_of_features = 128
batch_size = 55
batches_in_epoch = 1000
train_size = batches_in_epoch * batch_size
test_size = 10000
experiment_accuracy = [0, 0, 0]

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def printscreen(ndarrayinput, stringinput):
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

print("\n###################\nBuilding ConvNet to use as Feature Extractor\n###################")

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

W_fc1 = weight_variable([7*7*64, number_of_features])
b_fc1 = bias_variable([number_of_features])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([number_of_features, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

print("\n###################\nSVM Train/Test without Features from ConvNet\n###################")

converter = np.array([0,1,2,3,4,5,6,7,8,9])

train_features = np.zeros((train_size, 28*28), dtype=float)
train_labels = np.zeros(train_size, dtype=int)

for i in range(batches_in_epoch):
    train_batch = mnist.train.next_batch(batch_size)
    features_batch = train_batch[0]
    labels_batch = train_batch[1]
    for j in range(batch_size):
        for k in range(28*28):
            train_features[batch_size * i + j, k] = features_batch[j, k]
        train_labels[batch_size * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

printscreen(train_features, "train_features")

printscreen(train_labels, "train_labels")

test_features = np.zeros((test_size, 28*28), dtype=float)
test_labels = np.zeros(test_size, dtype=int)

test_features = mnist.test.images
for j in range(test_size):
    test_labels[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

printscreen(test_features, "test_features")

printscreen(test_labels, "test_labels")

clf = svm.SVC()
clf.fit(train_features, train_labels)
experiment_accuracy[0] = clf.score(test_features, test_labels)
print("\nSVM ACCURACY =", experiment_accuracy[0])

print("\n###################\nConvNet Train/Test for Features Extraction\n###################\n")

for i in range(4*batches_in_epoch):
    batch = mnist.train.next_batch(batch_size)
    if i%batches_in_epoch == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("epoch %d, training accuracy %g" % (i / batches_in_epoch, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

experiment_accuracy[1] = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("\nConvNet ACCURACY = %g" % experiment_accuracy[1])

print("\n###################\nSVM Train/Test with Features from ConvNet\n###################")

converter = np.array([0,1,2,3,4,5,6,7,8,9])

train_features_cnn = np.zeros((train_size, number_of_features), dtype=float)
train_labels_cnn = np.zeros(train_size, dtype=int)

for i in range(batches_in_epoch):
    train_batch = mnist.train.next_batch(batch_size)
    features_batch = h_fc1.eval(feed_dict={x: train_batch[0]})
    labels_batch = train_batch[1]
    for j in range(batch_size):
        for k in range(number_of_features):
            train_features_cnn[batch_size * i + j, k] = features_batch[j, k]
        train_labels_cnn[batch_size * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

printscreen(train_features_cnn, "train_features_cnn")

printscreen(train_labels_cnn, "train_labels_cnn")

test_features_cnn = h_fc1.eval(feed_dict={x: mnist.test.images})
test_labels_cnn = np.zeros(test_size, dtype=int)

for j in range(test_size):
    test_labels_cnn[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

printscreen(test_features_cnn, "test_features_cnn")

printscreen(test_labels_cnn, "train_labels_cnn")

clf = svm.SVC()
clf.fit(train_features_cnn, train_labels_cnn)
experiment_accuracy[2] = clf.score(test_features_cnn, test_labels_cnn)
print("\nConvNetSVM ACCURACY =", experiment_accuracy[2])

print("\n###################\nEnding Experiment\n###################\n")

print(experiment_accuracy)

print()

sess.close()
