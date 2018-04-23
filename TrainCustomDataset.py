# This is the code for training the custom dataset. 
# Here we read and train all the data which is stored as a TFRecord

import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join

CURRENT_DIR = "/home/linux/Desktop/CharacterTFRecord"
TRAINING_SET_SIZE = 6345
BATCH_SIZE = 36
IMAGE_SIZE = 28
CLASS_NUMBER = 62

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("image_number", 6345 , "Number of images in your tfrecord, default is 6345.")
flags.DEFINE_integer("class_number", 62, "Number of class in your dataset/label.txt, default is 3.")
flags.DEFINE_integer("image_height", 28, "Height of the output image after crop and resize. Default is 28.")
flags.DEFINE_integer("image_width", 28, "Width of the output image after crop and resize. Default is 28.")

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# image object from protobuf
class image_object:
    def __init__(self):
		self.image = tf.Variable([], dtype = tf.string)
		self.height = tf.Variable([], dtype = tf.int64)
		self.width = tf.Variable([], dtype = tf.int64)
		self.filename = tf.Variable([], dtype = tf.string)
		self.label = tf.Variable([], dtype = tf.int64)

# reading and decoding the tfrecord because its in binary
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    current_image_object = image_object()
    current_image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, FLAGS.image_height, FLAGS.image_width) # cropped image with size 28*28
    current_image_object.height = features["image/height"] # height of the raw image
    current_image_object.width = features["image/width"] # width of the raw image
    current_image_object.filename = features["image/filename"] # filename of the raw image
    current_image_object.label = tf.cast(features["image/class/label"], tf.float32) # label of the raw image
    return current_image_object

def flower_input(if_random = True, if_training = True):
    if(if_training):
        filenames = [os.path.join(CURRENT_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in xrange(0, 1)]
    else:
        filenames = [os.path.join(CURRENT_DIR, "validation-0000%d-of-00002.tfrecord" % i) for i in xrange(0, 1)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
	# image = image_object.image
	# image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
	return image_batch, label_batch, filename_batch

# method for training the custom data. 
# This operation will take a long time depending on the amount of train data
def customDataTrain():	
	with tf.Session() as sess:		
		batch = flower_input(if_random = False, if_training = True)
		x = batch[0]
		y_ = batch[1]
		
		#Layer 1
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		x_image = tf.reshape(x, [-1,28,28,1])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		#Layer 2
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

		#Densely Connected Layer
		W_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		#Dropout - reduz overfitting
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		#Readout layer
		W_fc2 = weight_variable([1024, 36])
		b_fc2 = bias_variable([36])
		y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		#Train and evaluate
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,-1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		#sess.run(tf.initialize_all_variables())

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)		
		for i in range(TRAINING_SET_SIZE):
		  if i%50 == 0:   
			train_accuracy = accuracy.eval(feed_dict={keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		  train_step.run(feed_dict={keep_prob: 0.5})
		
		saver = tf.train.Saver()
		saver.save(sess, CURRENT_DIR+"/trainsession.ckpt")
		sess.close()
		print("Finished Training Custom Dataset")

# method for validating the trained data
def customDatasetEvaluate():
	batch = flower_input(if_random = False, if_training = False)

	x = batch[0]
	y_ = batch[1]

	#Layer 1
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#Layer 2
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	#Densely Connected Layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#Dropout - reduz overfitting
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#Readout layer
	W_fc2 = weight_variable([1024, 36])
	b_fc2 = bias_variable([36])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	#Train and evaluate
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,-1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, CURRENT_DIR+"/trainsession.ckpt")
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess = sess)
		accuracy_accu = 0

		for i in range(29):
			accuracy_out, logits_batch_out = sess.run([accuracy, tf.to_int64(tf.arg_max(x, dimension = 1)))
			accuracy_accu += accuracy_out

			print(i)
			print(image_out.shape)
			print("label_out: ")
			print(filename_out)
			print(label_out)
			print(logits_batch_out)

		print("Accuracy: ")
		print(accuracy_accu / 29)

		coord.request_stop()
		coord.join(threads)
		sess.close()

customDataTrain()
#customDatasetEvaluate()

