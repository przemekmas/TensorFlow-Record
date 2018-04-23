# Copyright 2016 Google Inc. All Rights Reserved.
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

# The CreateTFRecord.py is used to create a TFRecord file which will contain images and corresponding labels
# The sub directories will contain images which correspond to the folder name
# Most of the code has been copied from https://github.com/yeephycho/tensorflow_input_image_by_tfrecord
# Code refactored and adapted by Przemyslaw Maslowski

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('image_format', 'JPEG', 'The colorspace')
tf.app.flags.DEFINE_string('colorspace', 'RGB', 'The colorspace')
tf.app.flags.DEFINE_string('train_directory', './train', 'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', './test', 'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', './', 'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 2, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 2, 'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('channels', 3, 'Specifies the amount of color channels')

# The labels file contains a list of valid labels are held in this file.
# Where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', './imagelabels', 'Labels file')

FLAGS = tf.app.flags.FLAGS

# Wrapper for inserting int64 features into Example proto.
def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Wrapper for inserting bytes features into Example proto.
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Build an Example proto for an example.
def _convert_to_example(filename, image_buffer, label, text, height, width):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height), # integer, image height in pixels
      'image/width': _int64_feature(width), # integer, image width in pixels
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(FLAGS.colorspace)), # colorspace
      'image/channels': _int64_feature(FLAGS.channels), # number of colour channels
      'image/class/label': _int64_feature(label), # integer, identifier for the ground truth for the network
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)), # string, unique human-readable, e.g. 'dog'
      'image/format': _bytes_feature(tf.compat.as_bytes(FLAGS.image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))), # string, path to an image file, e.g., '/path/to/example.JPG'
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))})) # string, JPEG encoding of RGB image

# Determine if a file contains a PNG format image.
def _is_png(filename):
  return '.png' in filename

# Process a single image file. Returns the following:
# image_buffer: string, JPEG encoding of RGB image.
# height: integer, image height in pixels.
# width: integer, image width in pixels.
def _process_image(filename, coder):
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]+8
  width = image.shape[1]+8
  assert image.shape[2] == 3

  return image_data, height, width

# Processes and saves list of images as TFRecord in 1 thread.
def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  """
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label, text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

# Process and save list of images as TFRecord of Example protos.
def _process_image_files(name, filenames, texts, labels, num_shards):
  """
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()

# Build a list of all images files and labels in the data set.
def _find_image_files(data_dir, labels_file):
  """
  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in JPEG files located in
      the following directory structure.

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg

      where 'dog' is the label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels

# Process a complete data set and save it as a TFRecord.
def _process_dataset(name, directory, num_shards, labels_file):
  """
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, texts, labels = _find_image_files(directory, labels_file)
  _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Process our dataset for validation and train images
  _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards, FLAGS.labels_file)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards, FLAGS.labels_file)

# Helper class that provides TensorFlow image coding utilities.
class ImageCoder(object):
  def __init__(self):    
    self._sess = tf.Session() # Create a single Session to run all image coding calls.

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=FLAGS.channels)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=FLAGS.channels)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
	
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

if __name__ == '__main__':
  tf.app.run()

