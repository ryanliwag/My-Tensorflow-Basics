from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from random import shuffle

import tensorflow as tf

##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128

def read_data(path):
  dataset = []
  for dirname, dirnames, filenames in os.walk(path):
    for filename in filenames:
      dataset.append([os.path.basename(dirname), os.path.join(dirname, filename)])

  shuffle(dataset)
  filenames = []
  labels = []
  for i in dataset:
    filenames.append(i[1])
    if i[0] == "cats":
      labels.append(0)
    else:
      labels.append(1)
  return filenames, labels

def _parse_function(filename, label):

  one_hot = tf.one_hot(label, NUM_CLASSES)

  img_file = tf.read_file(filename)
  img_decoded = tf.image.decode_jpeg(img_file, channels=3)
  image_resized = tf.image.resize_images(img_decoded, [28, 28])
  return image_resized, one_hot

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


#model
img_size=28
num_channels=3
NUM_CLASSES=2
batch_size=12

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=NUM_CLASSES,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))








#Data 

train_path = "/home/elements/Desktop/v-env/tensorflow/cats_dogs/dataset/train/"
test_path = "/home/elements/Desktop/v-env/tensorflow/cats_dogs/dataset/test/"

training_files_, training_labels = read_data(train_path)
testing_files, testing_labels = read_data(test_path)

training_files = tf.constant(training_files_)
training_labels = tf.constant(training_labels)
testing_files = tf.constant(testing_files)
testing_labels = tf.constant(testing_labels)

# create TensorFlow Dataset objects
tr_data = tf.contrib.data.Dataset.from_tensor_slices((training_files, training_labels))
tr_data = tr_data.map(_parse_function)
tr_data = tr_data.shuffle(buffer_size=10)
tr_data = tr_data.repeat()
tr_data = tr_data.batch(500)
val_data = tf.contrib.data.Dataset.from_tensor_slices((testing_files, testing_labels))
val_data = val_data.map(_parse_function)
val_data = val_data.repeat()
val_data = val_data.batch(500)

# create TensorFlow Iterator object
iterator = tf.contrib.data.Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(100):
    sess.run(training_init_op)
    elem = sess.run(next_element)
    sess.run(optimizer, feed_dict={x: elem[0], y_true: elem[1]})
    acc = sess.run(accuracy, feed_dict={x: elem[0], y_true: elem[1]})


    sess.run(validation_init_op)
    elem = sess.run(next_element)
    acc_val = sess.run(accuracy, feed_dict={x: elem[0], y_true: elem[1]})
    print(i, acc, acc_val)




