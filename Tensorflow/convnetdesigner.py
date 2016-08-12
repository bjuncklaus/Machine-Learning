import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import genfromtxt

# Parameters
learning_rate = 0.001
training_iters = 2
display_step = 1

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units
CALCULATE_TRAINING = True
CALCULATE_TESTING = False
MULTIPLE_TESTING = False
SAVE_OUTPUT = True

TIME_STAMPS = 10
NUM_AMPLITUDE = 64
# NUM_FEATURES = 66
NUM_FEATURES = NUM_AMPLITUDE + 66
NUM_LABELS = 64
N_CLASSES = NUM_LABELS * TIME_STAMPS
IMGS_EXTENSION = ".png"
IMGS_DIR = "imgs/"
LOG_DIR = "logs/"
OUTPUT_DIR = "output/"
DATASET_DIR = ""
# DATASET_DIR = "/Users/bjuncklaus/Dropbox/Machine Learning/Data/Tested/snrG19.6.ms/"
# DATASET_DIR = "/Users/bjuncklaus/Dropbox/Machine Learning/Data/Current/G29.7-0.3_cal.ms/All/"
DATASET_DIR = "/Users/bjuncklaus/Dropbox/Machine Learning/Data/Current/G29.7-0.3_cal.ms/SPW0/"
DATASET_AMPLITUDE_DIR = "/Users/bjuncklaus/Dropbox/Machine Learning/Data/With Amplitude/G29.7-0.3_cal.ms/SPW0/"
# DATASET_DIR = "/Users/bjuncklaus/NRAO/NRAO/Flagged/Bruno_2016_06_10/"
# DATASET_DIR = "/users/bmartins/datasets/"
# DATASET_DIR = "/home/vega2/bmartins/datasets/"
# DATASET_DIR = "/Users/rurvashi/ForTensorFlow/Tensorflow/Data/"
ORIGINAL_DATA_FILENAME = "original_data.csv"
TEST_DATA_FILENAME = "testing.csv"
ORIGINAL_OUTPUT_FILENAME = "original_output.txt"
TESTING_OUTPUT_FILENAME = "testing_output.txt"


###################################

def show_graph(prediction, original_data):
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)

    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)

    plt.plot(original_data, color="blue", linewidth=2.0, linestyle="-")
    plt.plot(prediction, color="red", linewidth=2.0, linestyle="--")

    plt.show()

def remove_previous_imgs():
    filelist = [f for f in os.listdir(IMGS_DIR)]
    for f in filelist:
        os.remove(IMGS_DIR + f)

def save_image(data, img_name, iter, subplot=None, feature=False):
    if feature:
        for i in data[0]:
            img_name = time.strftime("%Y%m%d-%H%M%S")
            plot_feature(i)
            plt.savefig(IMGS_DIR + img_name + IMGS_EXTENSION)
    else:
        plot_img(data, iter, subplot=subplot)
        plt.savefig(IMGS_DIR + img_name + IMGS_EXTENSION)

def plot_feature(feature, show_plot=False):
    plt.clf()
    plt.subplot(121)
    plt.imshow(feature)

    if show_plot:
        plt.show()

def get_batch_flags(source):
    batch = []
    batch.append([])

    batch_index = 0
    i = 0
    for j in range(len(source)):
        if (j == i + NUM_LABELS):
            batch.append([])
            i += NUM_LABELS
            batch_index += 1
        batch[batch_index].append(source[j])

    batch = np.array(batch)
    return batch

data_amplitude = genfromtxt(DATASET_AMPLITUDE_DIR + ORIGINAL_DATA_FILENAME, delimiter=',', dtype='float')
current_polarization = 0

def get_amplitude():
    global current_polarization

    batch = []

    i = 0
    for j in range(TIME_STAMPS):
        batch.append([])

        for b in data_amplitude[j+current_polarization][NUM_FEATURES:NUM_FEATURES+NUM_AMPLITUDE]:
            batch[j].append(b)

    batch = np.array(batch)

    current_polarization += 10
    return batch

def plot_img(prediction, iter, show_plot=False, subplot=None):
    batch_flags = get_batch_flags(prediction[0])
    batch_labels_flags = get_batch_flags(subplot[0])

    batch = get_amplitude()
    print("BATCH", batch_labels_flags.shape)

    plt.clf()
    plt.subplot(121)
    # plt.xticks(np.arange(0, len(p[0]), 2), "CURRENT ITERATION: " + str(iter))
    plt.imshow(1-np.sqrt(batch) * np.abs(1-batch_flags), vmin=np.min(batch), vmax=1)

    plt.subplot(122)
    plt.imshow(batch_labels_flags)


    if (show_plot):
        plt.show()

def write_log(log_file, log):
    log_file.write(str(log))
    log_file.write("\n")

def write_output(filename, output):
    output_file = open(filename, "w")
    for value in output:
        output_file.write(str(value))
        output_file.write(",")
    output_file.close()

def read_lines_data(filename):
    text_file = open(filename, "r")
    lines = text_file.read().split(',')
    lines.remove('')
    text_file.close()

    return lines

def get_output(filename):
    lines = read_lines_data(OUTPUT_DIR + filename)

    for i in range(len(lines)):
        lines[i] = np.float32(lines[i])
    p = []
    p.append(lines)
    p = tf.Variable(np.array(p))


    return p

print(":: READING DATA ::")
if (CALCULATE_TRAINING):
    data = genfromtxt(DATASET_DIR + ORIGINAL_DATA_FILENAME, delimiter=',', dtype='float')

test_data = genfromtxt(DATASET_DIR + TEST_DATA_FILENAME, delimiter=',', dtype='float')
print(":: Finished ::")

# create_flag_all_data(test_data)
# print("ALL DATA", np.array(flag_all_data).shape)
remove_previous_imgs()

# Labels
y = tf.placeholder(tf.float32, [None, N_CLASSES])

# Dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)

""" 66x89x1 """

# x0 = tf.placeholder(tf.float32, shape=[None, 5874], name='Input')
#
# x1 = tf.reshape(x0, [-1,66,89,1])
# x2 = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
# x3 = tf.nn.conv2d(x1, x2, strides=[1, 3, 3, 1], padding='SAME')
#
# x4 = tf.nn.max_pool(x3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
# x5 = tf.Variable(tf.constant(0.1, shape=[32]))
# x6 = tf.nn.relu(x4 + x5)
#
# x7 = tf.Variable(tf.truncated_normal([21120, 10000], stddev=0.1))
# x8 = tf.reshape(x6, [-1, 21120])
# x9 = tf.matmul(x8, x7)
#
# y_conv = tf.nn.dropout(x9, dropout) # Apply droput before the output layer is created
#
# x10 = tf.Variable(tf.truncated_normal([10000, 5696], stddev=0.1))
# x11 = tf.reshape(x9, [-1, 10000])
# y_conv = tf.matmul(x11, x10)


""" 58x10x1 """
# x0 = tf.placeholder(tf.float32, shape=[None, 580])
# x0_ = tf.placeholder(tf.float32, shape=[None, 580])
#
# if (CALCULATE_TRAINING):
#     x1 = tf.reshape(x0, [-1,58,10,1])
#     x2 = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
#     x3 = tf.nn.conv2d(x1, x2, strides=[1, 2, 2, 1], padding='VALID')
#
#     x4 = tf.nn.max_pool(x3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x5 = tf.Variable(tf.constant(0.1, shape=[32]))
#     x6 = tf.nn.sigmoid(x4 + x5)
#
#     x7 = tf.reshape(x6, [-1,29,5,32])
#     x8 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
#     x9 = tf.nn.conv2d(x7, x8, strides=[1, 2, 2, 1], padding='VALID')
#
#     x10 = tf.nn.max_pool(x9, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x11 = tf.Variable(tf.constant(0.1, shape=[64]))
#     x12 = tf.nn.sigmoid(x10 + x11)
#
#     x13 = tf.reshape(x12, [-1,14,2,64])
#     x14 = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))
#     x15 = tf.nn.conv2d(x13, x14, strides=[1, 2, 2, 1], padding='SAME')
#
#     x16 = tf.nn.max_pool(x15, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x17 = tf.Variable(tf.constant(0.1, shape=[128]))
#     x18 = tf.nn.sigmoid(x16 + x17)
#
#     x19 = tf.Variable(tf.truncated_normal([896, 62500], stddev=0.1))
#     x20 = tf.reshape(x18, [-1, 896])
#     x21 = tf.matmul(x20, x19)
#
#     x22 = tf.Variable(tf.constant(0.1, shape=[1]))
#     x23 = tf.nn.sigmoid(x21 + x22)
#
#     x23 = tf.nn.dropout(x23, dropout) # Apply droput before the output layer is created
#
#     x24 = tf.Variable(tf.truncated_normal([62500, 560], stddev=0.1))
#     x25 = tf.reshape(x23, [-1, 62500])
#     y_conv = tf.matmul(x25, x24)
#
#     pred = y_conv
#
# else:
#     pred = get_output(ORIGINAL_OUTPUT_FILENAME)
#
# if (CALCULATE_TESTING):
#     x1_ = tf.reshape(x0_, [-1, 58, 10, 1])
#     x2_ = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
#     x3_ = tf.nn.conv2d(x1_, x2_, strides=[1, 2, 2, 1], padding='VALID')
#
#     x4_ = tf.nn.max_pool(x3_, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x5_ = tf.Variable(tf.constant(0.1, shape=[32]))
#     x6_ = tf.nn.sigmoid(x4_ + x5_)
#
#     x7_ = tf.reshape(x6_, [-1, 29, 5, 32])
#     x8_ = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
#     x9_ = tf.nn.conv2d(x7_, x8_, strides=[1, 2, 2, 1], padding='VALID')
#
#     x10_ = tf.nn.max_pool(x9_, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x11_ = tf.Variable(tf.constant(0.1, shape=[64]))
#     x12_ = tf.nn.sigmoid(x10_ + x11_)
#
#     x13_ = tf.reshape(x12_, [-1, 14, 2, 64])
#     x14_ = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))
#     x15_ = tf.nn.conv2d(x13_, x14_, strides=[1, 2, 2, 1], padding='SAME')
#
#     x16_ = tf.nn.max_pool(x15_, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x17_ = tf.Variable(tf.constant(0.1, shape=[128]))
#     x18_ = tf.nn.sigmoid(x16_ + x17_)
#
#     x19_ = tf.Variable(tf.truncated_normal([896, 62500], stddev=0.1))
#     x20_ = tf.reshape(x18_, [-1, 896])
#     x21_ = tf.matmul(x20_, x19_)
#
#     x22_ = tf.Variable(tf.constant(0.1, shape=[1]))
#     x23_ = tf.nn.sigmoid(x21_ + x22_)
#
#     x23_ = tf.nn.dropout(x23_, dropout)  # Apply droput before the output layer is created
#
#     x24_ = tf.Variable(tf.truncated_normal([62500, 560], stddev=0.1))
#     x25_ = tf.reshape(x23_, [-1, 62500])
#     y_conv_ = tf.matmul(x25_, x24_)
#
#     pred_ = y_conv_
# else:
#     pred_ = get_output(TESTING_OUTPUT_FILENAME)

""" 64x10x1 """
# x0 = tf.placeholder(tf.float32, shape=[None, 660])
# x0_ = tf.placeholder(tf.float32, shape=[None, 660])
#
# if (CALCULATE_TRAINING):
#     x1 = tf.reshape(x0, [-1,66,10,1])
#     x2 = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
#     x3 = tf.nn.conv2d(x1, x2, strides=[1, 2, 2, 1], padding='VALID')
#
#     x4 = tf.nn.max_pool(x3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x5 = tf.Variable(tf.constant(0.1, shape=[32]))
#     x6 = tf.nn.relu(x4 + x5)
#
#     x7 = tf.reshape(x6, [-1,33,5,32])
#     x8 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
#     x9 = tf.nn.conv2d(x7, x8, strides=[1, 2, 2, 1], padding='VALID')
#
#     x10 = tf.nn.max_pool(x9, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x11 = tf.Variable(tf.constant(0.1, shape=[64]))
#     x12 = tf.nn.relu(x10 + x11)
#
#     x13 = tf.reshape(x12, [-1,16,2,64])
#     x14 = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))
#     x15 = tf.nn.conv2d(x13, x14, strides=[1, 2, 2, 1], padding='VALID')
#
#     x16 = tf.nn.max_pool(x15, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x17 = tf.Variable(tf.constant(0.1, shape=[128]))
#     x18 = tf.nn.relu(x16 + x17)
#
#     x19 = tf.Variable(tf.truncated_normal([1024, 90000], stddev=0.1))
#     x20 = tf.reshape(x18, [-1, 1024])
#     x21 = tf.matmul(x20, x19)
#
#     x22 = tf.Variable(tf.constant(0.1, shape=[1]))
#     x23 = tf.nn.relu(x21 + x22)
#
#     x23 = tf.nn.dropout(x23, dropout)  # Apply droput before the output layer is created
#
#     x24 = tf.Variable(tf.truncated_normal([90000, 640], stddev=0.1))
#     x25 = tf.reshape(x23, [-1, 90000])
#     y_conv = tf.matmul(x25, x24)
#
#     pred = y_conv
# else:
#     pred = get_output(ORIGINAL_OUTPUT_FILENAME)
#
# if (CALCULATE_TESTING):
#     x1_ = tf.reshape(x0_, [-1,66,10,1])
#     x2_ = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
#     x3_ = tf.nn.conv2d(x1_, x2_, strides=[1, 2, 2, 1], padding='VALID')
#
#     x4_ = tf.nn.max_pool(x3_, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x5_ = tf.Variable(tf.constant(0.1, shape=[32]))
#     x6_ = tf.nn.relu(x4_ + x5_)
#
#     x7_ = tf.reshape(x6_, [-1,33,5,32])
#     x8_ = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
#     x9_ = tf.nn.conv2d(x7_, x8_, strides=[1, 2, 2, 1], padding='VALID')
#
#     x10_ = tf.nn.max_pool(x9_, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x11_ = tf.Variable(tf.constant(0.1, shape=[64]))
#     x12_ = tf.nn.relu(x10_ + x11_)
#
#     x13_ = tf.reshape(x12_, [-1,16,2,64])
#     x14_ = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))
#     x15_ = tf.nn.conv2d(x13_, x14_, strides=[1, 2, 2, 1], padding='VALID')
#
#     x16_ = tf.nn.max_pool(x15_, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     x17_ = tf.Variable(tf.constant(0.1, shape=[128]))
#     x18_ = tf.nn.relu(x16_ + x17_)
#
#     x19_ = tf.Variable(tf.truncated_normal([1024, 90000], stddev=0.1))
#     x20_ = tf.reshape(x18_, [-1, 1024])
#     x21_ = tf.matmul(x20_, x19_)
#
#     x22_ = tf.Variable(tf.constant(0.1, shape=[1]))
#     x23_ = tf.nn.relu(x21_ + x22_)
#
#     x23_ = tf.nn.dropout(x23_, dropout)  # Apply droput before the output layer is created
#
#     x24_ = tf.Variable(tf.truncated_normal([90000, 640], stddev=0.1))
#     x25_ = tf.reshape(x23_, [-1, 90000])
#     y_conv_ = tf.matmul(x25_, x24_)
#
#     pred_ = y_conv_
# elif MULTIPLE_TESTING:
#     i = 3
#     t = []
#     t.append(np.float32(read_lines_data(OUTPUT_DIR + str(i) + ".txt")))
#     pred_ = tf.Variable(np.array(t))
# else:
#     pred_ = get_output(TESTING_OUTPUT_FILENAME)

""" 130x10x1 """
x0 = tf.placeholder(tf.float32, shape=[None, 1300])
x0_ = tf.placeholder(tf.float32, shape=[None, 1300])

if (CALCULATE_TRAINING):
    x1 = tf.reshape(x0, [-1, 130, 10, 1])
    x2 = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
    x3 = tf.nn.conv2d(x1, x2, strides=[1, 2, 2, 1], padding='VALID')

    x4 = tf.nn.max_pool(x3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    x5 = tf.Variable(tf.constant(0.1, shape=[32]))
    x6 = tf.nn.relu(x4 + x5)

    x7 = tf.reshape(x6, [-1, 65, 5, 32])
    x8 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.1))
    x9 = tf.nn.conv2d(x7, x8, strides=[1, 3, 3, 1], padding='VALID')

    x10 = tf.nn.max_pool(x9, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    x11 = tf.Variable(tf.constant(0.1, shape=[64]))
    x12 = tf.nn.relu(x10 + x11)

    x13 = tf.reshape(x12, [-1, 22, 2, 64])
    x14 = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))
    x15 = tf.nn.conv2d(x13, x14, strides=[1, 2, 2, 1], padding='VALID')

    x16 = tf.nn.max_pool(x15, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    x17 = tf.Variable(tf.constant(0.1, shape=[128]))
    x18 = tf.nn.relu(x16 + x17)

    x19 = tf.Variable(tf.truncated_normal([1408, 90000], stddev=0.1))
    x20 = tf.reshape(x18, [-1, 1408])
    x21 = tf.matmul(x20, x19)

    x22 = tf.Variable(tf.constant(0.1, shape=[1]))
    x23 = tf.nn.relu(x21 + x22)

    x23 = tf.nn.dropout(x23, dropout)  # Apply droput before the output layer is created

    x24 = tf.Variable(tf.truncated_normal([90000, 640], stddev=0.1))
    x25 = tf.reshape(x23, [-1, 90000])
    y_conv = tf.matmul(x25, x24)

    pred = y_conv
    pred_ = y_conv
else:
    pred = get_output(ORIGINAL_OUTPUT_FILENAME)
    pred_ = get_output(ORIGINAL_OUTPUT_FILENAME)

# TODO - Fix the COST and ACCURACY
print(":: Calculating ACCURACY ::")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
n = 1
bias = n * tf.sqrt(tf.constant(1.0/TIME_STAMPS*NUM_LABELS) * tf.reduce_sum(tf.square(pred)))
# bias_ = n * tf.sqrt(tf.constant(1.0/TIME_STAMPS*NUM_LABELS) * tf.reduce_sum(tf.square(tf.reduce_mean(pred_))))
# bias_ = tf.constant(1, dtype=tf.float32)
bias_ = tf.constant(0, dtype=tf.float32)

mean = tf.reduce_mean(pred)
mean_ = tf.reduce_mean(pred_)

# division = tf.truediv(pred, mean)
division = pred
# division_ = tf.truediv(pred, mean)
division_ = tf.truediv(pred, pred) * 100
# division_ = tf.add(pred, pred_)
# division_ = tf.truediv(pred_, pred)

predicted = tf.cast(tf.less_equal(bias, division), tf.float32)
predicted_ = tf.cast(tf.less_equal(bias_, division_), tf.float32)
# predicted_ = tf.cast(tf.less_equal(bias_, pred_), tf.float32)

predicted_converted = tf.cast(predicted, tf.float32)
predicted_converted_ = tf.cast(predicted_, tf.float32)

correct_pred = tf.equal(predicted, y)
correct_pred_ = tf.equal(predicted_, y)

correct_pred_converted = tf.cast(correct_pred, tf.float32)
correct_pred_converted_ = tf.cast(correct_pred_, tf.float32)

accuracy = tf.reduce_mean(correct_pred_converted)
accuracy_ = tf.reduce_mean(correct_pred_converted_)

print(":: Finished ::")

# Initializing the variables
init = tf.initialize_all_variables()

print(":: LAUNCHING Graph ::")
with tf.Session() as sess:
    print(":: SESSION RUN ::")
    sess.run(init)
    step = 1

    if (CALCULATE_TRAINING):
        x_train = []
        x_train_buffer = []
        y_train_onehot = []
        y_train_onehot_buffer = []
        total_lines_data = len(data)
        for current_line in range(total_lines_data):
            for i in data[current_line][0:NUM_FEATURES]:
                x_train_buffer.append(i)

            for i in data[current_line][NUM_FEATURES:]:
                y_train_onehot_buffer.append(i)

            if (len(y_train_onehot_buffer) == NUM_LABELS*TIME_STAMPS):
                y_train_onehot.append(np.array(y_train_onehot_buffer))
                y_train_onehot_buffer = []

            if (len(x_train_buffer) == NUM_FEATURES*TIME_STAMPS):
                x_train.append(np.array(x_train_buffer))
                x_train_buffer = []

        x_train = np.array(x_train)
        y_train_onehot = np.array(y_train_onehot)

    x_test = []
    x_test_buffer = []
    y_test_onehot = []
    y_test_onehot_buffer = []
    total_lines_test_data = len(test_data)
    for current_line in range(total_lines_test_data):
        for i in test_data[current_line][0:NUM_FEATURES]:
            x_test_buffer.append(i)

        for i in test_data[current_line][NUM_FEATURES:]:
            y_test_onehot_buffer.append(i)

        if (len(y_test_onehot_buffer) == NUM_LABELS*TIME_STAMPS):
            y_test_onehot.append(np.array(y_test_onehot_buffer))
            y_test_onehot_buffer = []

        if (len(x_test_buffer) == NUM_FEATURES*TIME_STAMPS):
            x_test.append(np.array(x_test_buffer))
            x_test_buffer = []

    x_test = np.array(x_test)
    y_test_onehot = np.array(y_test_onehot)

    global_iteration = 0
    while step <= training_iters:
        print("Step:", step)

        # print("x_train shape=", x_train.shape)
        # print("y_train_onehot shape=", y_train_onehot.shape)

        if (CALCULATE_TRAINING):
            length_x_train = len(x_train)
            for i in range(length_x_train):
                # Run optimization op (backprop)
                train = np.array([x_train[i]])
                label = np.array([y_train_onehot[i]])

                _, p, c, m, d, pc = sess.run([optimizer, pred, correct_pred, mean, division, predicted], feed_dict={y: label, keep_prob: dropout, x0: train})
                # print("PREDIC CONVERTED: ", pc)
                # print("PERCENTAGE PREDS =", np.array(p / np.max(p)))
                # print("PRED =", p.shape)
                # print("X =", x.shape)
                print("M =", m)


                print("Global Iter=", global_iteration, "| Training Iter=", i,
                      "| Total Trained =", "{:.2f}".format(((float(global_iteration)/float(length_x_train)) / float(training_iters)) * 100.0)+"%")

                # file_name = time.strftime("%Y%m%d-%H%M%S")
                # save_image(p, file_name, global_iteration, feature=True)
                # save_image(p, file_name, global_iteration)

                global_iteration += 1
                # break
            if (SAVE_OUTPUT):
                write_output(OUTPUT_DIR + ORIGINAL_OUTPUT_FILENAME, p[0])

        #if (c == True):
          #  print("PRED=", p)
            # print("TRAIN", x_train)

        # sess.run(pred)

        mean_acc = 0
        mean_acc_ = 0
        if step % display_step == 0:
            length_x_test = len(x_test)
            log_filename = LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + "log.txt"
            log_file = open(log_filename, "w")
            for i in range(length_x_test):
                label = np.array([y_test_onehot[i]])
                test = np.array([x_test[i]])
                # print("test ", test.shape, test)
                # print("label ", label.shape, label)

                # Calculate batch loss and accuracy
                pc, loss, acc, cpc_, cpc, b, m, pc_, acc_, d_, b_, m_, p_, p, d = sess.run([predicted_converted, cost, accuracy, correct_pred_converted_, correct_pred_converted, bias, mean, predicted_converted_, accuracy_, division_, bias_, mean_, pred_ , pred, division], feed_dict={x0: test, y: label, keep_prob: 1.})

                print("Bias: ", b, "Mean:", m, "Bias_:", b_, "Mean_:", m_)
                # print("DIV_: ", d_)
                # print("PREDIC CONV_: ", pc_)

                # print("Accuracy Converted:", acc_converted)
                log1 = "Test Iteration:", i, "| Step:", step, "| Training Accuracy=", "{:.5f}".format(acc * 100.0) + "%", "| MY Accuracy=", "{:.5f}".format(acc_ * 100.0) + "%"
                print(log1)
                write_log(log_file, log1)
                mean_acc += acc
                mean_acc_ += acc_

                file_name = time.strftime("%Y%m%d-%H%M%S")
                # if (acc > acc_):
                #     data = pc
                # else:

                """ HERE """
                # temp = []
                # temp.append([])
                # for j in p_[0]:
                #     if (j < 0):
                #         temp[0].append(-1)
                #         continue
                #     temp[0].append(1)
                #
                # temp = np.array(temp)

                """ HERE """
                # data = (p / np.sum(p) * 100) + (temp * (np.abs(p_) / np.sum(np.abs(p_)) * 100))
                # data = (p / np.sum(p) * 100) * (np.abs(p_) / np.sum(np.abs(p_)) * 100)
                # data = (p / np.sum(p) * 100)
                # data = (p / (np.sum(p)+np.sum(p_)) * 100) + (p_ / (np.sum(p)+np.sum(p_)) * 100)

                """ HERE """
                # t = 1
                # print("AVG:", np.mean(data), "A_:", np.mean(data)*t)
                # data = np.int8(data > np.mean(data)*t)
                """ HERE """

                # flag_sobreposition = 1-(data * p)
                # flag_sobreposition = (data * p) > 0
                # flag_sobreposition = np.int8(flag_sobreposition)

                """ HERE """
                # save_image(data, file_name+"__", i, label)
                save_image(pc, file_name, i, label)
                # save_image(pc_, file_name+"_", i, label)
                # save_image(p, file_name+"___", i, label)
                # save_image(p_, file_name+"____", i, label)
                """ HERE """
                # break
                # print("Test Iteration:", i, "| Step:", step, "| Minibatch Loss=" , "{:.6f}".format(loss), "| Training Accuracy=", "{:.5f}".format(acc))
                # plot_img(acc_converted, global_iteration, show_plot=True)

                if (SAVE_OUTPUT):
                    if (MULTIPLE_TESTING):
                        write_output(OUTPUT_DIR + str(i) + ".txt", p_[0])
                    else:
                        write_output(OUTPUT_DIR + TESTING_OUTPUT_FILENAME, p_[0])

            log_file.close()
            print("BIAS =", b)

            # file_name = time.strftime("%Y%m%d-%H%M%S")
            # save_image(acc_converted, file_name, global_iteration)

            print("Mean Accuracy:",  "{:.2f}".format(100.0 * (mean_acc / length_x_test))+"%")
            print("MY Mean Accuracy:",  "{:.2f}".format(100.0 * (mean_acc_ / length_x_test))+"%")

        step += 1
    print("Optimization Finished!")