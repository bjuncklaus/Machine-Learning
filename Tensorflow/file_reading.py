import tensorflow as tf

data_dir = "/Users/bjuncklaus/NRAO/NRAO/Flagged/Bruno_2016_06_10/data.csv"
label_dir = "/Users/bjuncklaus/NRAO/NRAO/Flagged/Bruno_2016_06_10/label.csv"

filename_queue = tf.train.string_input_producer([data_dir, label_dir])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# record_defaults = [[1], [1], [1], [1], [1]]
record_defaults = []

for _ in range(5697):
    record_defaults.append(['0'])
# col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
cols = tf.decode_csv(value, record_defaults=record_defaults)
# features = tf.pack([col1, col2, col3, col4])
features = tf.pack([cols])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a single instance:
        # example, label = sess.run([features, col5])
        example, label = sess.run([features, cols[4]])
        # print("Example", example)
        # print("Label", label)


    coord.request_stop()
    coord.join(threads)