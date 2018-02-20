""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os
import scipy.io.wavfile as wav
import numpy as np
import random
from utils import pad_sequences as pad_sequences
from six.moves import xrange as range

try:
    from python_speech_features import mfcc, logfbank
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

# Import MNIST data
from utils import sparse_tuple_from as sparse_tuple_from


SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

num_features = 13
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 2000
#num_epochs = 2000
# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10
# Loading the data
audioFN = list()
monsterDict = dict()
audioValFN = list()
lenDict = dict()

# tf Graph input
X = tf.placeholder(tf.float32, [ None, None, num_features])
Y = tf.sparse_placeholder(tf.int32 )
seq_len = tf.placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Loading the data
audioFN = list()
monsterDict = dict()
audioValFN = list()
#ll = os.listdir('data/processed_files/')
with open('trial_list_3_sec','r') as fp:
  cc = fp.readlines()

audioFN = [x.strip('\n') for x in cc]

with open('three-sec-transcription.txt','r') as fp:
  cc1 = fp.readlines()

textDict = dict()
neoaudioFN = list()
for elem in cc1:
  loc = elem.strip('\n').split('#')
  if loc[0] in audioFN:
    textDict[ loc[0] ] = loc[1]
    neoaudioFN.append( loc[0] )

ll = os.listdir('data/processed_files/valid')
for fn in ll:
  if '.wav' in fn:
    audioValFN.append( 'data/processed_files/valid/'+fn.split('.')[0] )
num_batches_per_epoch = max(int(len(neoaudioFN)/batch_size), 1 )

original = ''
batch_train_ip      = list()
batch_train_tgt     = list()
batch_val_ip      = list()
batch_val_tgt     = list()
batch_train_seq_len = list()

def retPro( ln ):
  z1 = ln.replace('0','zero ')
  z2 = z1.replace('1','one ')
  z3 = z2.replace('2','two ')
  z4 = z3.replace('3','three ')
  z5 = z4.replace('4','four ')
  z6 = z5.replace('5','five ')
  z7 = z6.replace('6','six ')
  z8 = z7.replace('7','seven ')
  z9 = z8.replace('8','eight ')
  z10 = z9.replace('9','nine ')
  return z10

def retFiles( fname, ctr , mode='NA'):
  if mode == 'VAL':
    fs, audio = wav.read( fname+'.wav' , ctr )
  else:
    fs, audio = wav.read( 'data/onemore/'+fname , ctr )
  #inputs = logfbank(audio, samplerate=fs)
  inputs = mfcc(audio, samplerate=fs)
# Tranform in 3D array
  train_inputs = inputs
  #train_inputs = np.asarray(inputs[np.newaxis, :])
  train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
  train_seq_len = [train_inputs.shape[1]]
  targets = []
# Readings targets
  if ctr >= 0:
    if mode == 'VAL':
      with open( fname+'.txt', 'r') as f:
        line = f.readlines()[-1]
      proL = retPro( line )
    else:
      proL = textDict[ fname ]
    original = ' '.join(proL.strip().lower().split(' ')[2:]).replace('.', '')
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

# Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
    hapur = targets
# Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                      for x in targets])
# Creating sparse representation to feed the placeholder
  else:
    train_targets = []
  if len(train_inputs) == 299 :
    monsterDict[ fname ] = { 'inp': train_inputs  ,'tgt': targets, 'len': len(train_inputs) }
  return train_inputs, train_targets, train_seq_len

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    print('HOLA '+str(x))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 299 , num_features , 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([10*10*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
batch_s = tf.shape(X)[0]
# Construct model
logits = conv_net(X, weights, biases, keep_prob)
# Define loss and optimizer
logits = tf.reshape(logits, [batch_size, -1, num_classes])

logits = tf.transpose(logits, (1, 0, 2))

loss = tf.nn.ctc_loss(Y , logits, seq_len)

cost = tf.reduce_mean(loss)
optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def return_next_batch( batch_sz ):
  yy = monsterDict.keys()
  random.shuffle( yy )
  batch_tgt = list()
  batch_ip = list()
  batch_len = list()
  for elem in range(len(yy)):
    if elem > batch_sz-1:
      batch_ip = np.asarray( batch_ip )
      batch_len = np.asarray( batch_len )
      batch_tgt = sparse_tuple_from( batch_tgt )
      return batch_ip, batch_tgt, batch_len
    #hold = np.asarray( tf.reshape( monsterDict[ yy[elem] ]['inp'], [ monsterDict[ yy[elem] ]['len'] , num_features ] ) )

    batch_ip.append( (monsterDict[ yy[elem] ]['inp']) )
    batch_tgt.append( monsterDict[ yy[elem] ]['tgt'] )
    batch_len.append( monsterDict[ yy[elem] ]['len'] )

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y, batch_z = return_next_batch(batch_size)
        # Run optimization op (backprop)
        b_cost, _ = sess.run( [cost, optimizer], feed_dict={X:  batch_x, Y: batch_y, seq_len: batch_z, keep_prob:0.5 } )
