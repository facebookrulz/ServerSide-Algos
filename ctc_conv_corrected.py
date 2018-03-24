#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range

try:
    from python_speech_features import mfcc, logfbank
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences

#import tensorflow.contrib.eager as tfe

#tfe.enable_eager_execution()
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 26
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 1000
#num_epochs = 100
num_hidden = 256
num_norm_layers = 4
num_resi_layers = 10
batch_size = 32
initial_learning_rate = 1e-5
momentum = 0.9


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Loading the data
monsterDict = dict()
audioValFN = list()
#ll = os.listdir('data/processed_files/')

with open('vox/gg.txt','r') as fp:
  cc = fp.readlines()

chekL = [c.strip('\n') for c in cc]

with open('vox/keyFile.txt','r') as fp:
  cc1 = fp.readlines()


textDict = dict()
neoaudioFN = list()
for elem in cc1:
  loc = elem.strip('\n').split('#')
  if loc[0] in chekL:
    textDict[ loc[0] ] = loc[1]
    neoaudioFN.append( loc[0] )

print('OOOOOOOOOOOLLLLLLLLLLLLLLLLLLL   '+str(len(neoaudioFN)))
print('OOOOOOOOOOOLLLLLLLLLLLLLLLLLLL   '+str(len(textDict)))
num_batches_per_epoch = max(int(len(neoaudioFN)/batch_size), 1 )
num_examples = len(neoaudioFN)
print( num_batches_per_epoch )
print( num_examples )
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

def get_a_cell(lstm_size, keep_prob, mode):
  lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
  if mode == 'RESI':
    lstm = tf.contrib.rnn.ResidualWrapper( lstm )
    return lstm
  return drop

def retFiles( fname, ctr , mode='NA'):
  if mode == 'VAL':
    fs, audio = wav.read( fname+'.wav' , ctr )
  else:
    fs, audio = wav.read( 'vox/'+fname+'.wav' , ctr )
  inputs = logfbank(audio, samplerate=fs)
  #inputs = mfcc(audio, samplerate=fs)
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
    for elem in targets:
      if elem < 0:
        print('emir')
        print(targets)
        print(hapur)
# Creating sparse representation to feed the placeholder
    #train_targets = sparse_tuple_from([targets])
    train_targets = (targets)
  else:
    train_targets = []
  monsterDict[ fname ] = {'inp': train_inputs  ,'tgt': train_targets, 'len': train_seq_len}
  return train_inputs, train_targets, train_seq_len


for ctr in range(len(neoaudioFN)):
  train_inputs, train_targets, train_seq_len = retFiles( neoaudioFN[ctr] , ctr )
  batch_train_ip.append( train_inputs )
  batch_train_tgt.append( monsterDict[ neoaudioFN[ctr] ]['tgt'] )

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# You can preprocess the input data here
train_inputs = np.asarray( batch_train_ip )

# You can preprocess the target data here
train_targets = np.asarray( batch_train_tgt )


# THE MAIN CODE!
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

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([10, 10,  1, num_classes])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([8, 8, num_classes, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([num_features*num_classes, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([num_classes])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# tf Graph input
X = tf.placeholder(tf.float32, [ None, None, num_features])
Y = tf.sparse_placeholder(tf.int32 )
seq_len = tf.placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

batch_s = tf.shape(X)[0]
# Construct model

x = tf.reshape(X, shape=[-1, num_features, 399 , 1])
    # Convolution Layer
conv1 = conv2d(conv1, weights['wc1'], biases['bc1'], 1)
print('xxxxxxxxxxxxxxxx')
fc1 = tf.reshape(conv1, [batch_size , 399 , weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.layers.dense( fc1, 1024 , activation=tf.nn.relu)
#print('xxxxxxxxxxxxxxxx')
#print(fc1)
#fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#print('xxxxxxxxxxxxxxxx')
#print(fc1)
    # Apply Dropout
fc1 = tf.nn.dropout(fc1, keep_prob)
    # Output, class prediction
logits = tf.layers.dense(inputs=fc1, units=num_classes, activation=tf.nn.relu)

logits = tf.transpose(logits, (1, 0, 2))

loss = tf.nn.ctc_loss(Y , logits, seq_len)

cost = tf.reduce_mean(loss)
optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)
    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          Y))


with tf.Session() as session:
    saver = tf.train.Saver()
    # Initializate the weights and biases
    tf.global_variables_initializer().run()
    #saver.restore( session, '/home/ec2-user/speech/tf/resi_save/baby_steps-91' )
    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()
        if curr_epoch!=0 and curr_epoch%15 == 0:
          initial_learning_rate = initial_learning_rate * 10
          momentum = max(0.99, momentum*1.05)
        for batch in range(num_batches_per_epoch):

            # Getting the index
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

            batch_train_inputs = train_inputs[indexes]
            # Padding input to max_time_step of this batch
            batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
            # Converting to sparse representation so as to to feed SparseTensor input
            batch_train_targets = sparse_tuple_from(train_targets[indexes])

            feed = {X: batch_train_inputs,
                    Y: batch_train_targets,
                    seq_len: batch_train_seq_len,
                    keep_prob:0.5}

            ll, ll1, ll2 = session.run([conv1, fc1, logits], feed)
            print( np.asarray(ll).shape )
            print( np.asarray(ll1).shape )
            print( np.asarray(ll2).shape )
            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size


        # Shuffle the data
        shuffled_indexes = np.random.permutation(num_examples)
        train_inputs = train_inputs[shuffled_indexes]
        train_targets = train_targets[shuffled_indexes]

        # Metrics mean
        train_cost /= num_examples
        train_ler /= num_examples

        saved_path = saver.save( session, 'resi_save/baby_steps', global_step=curr_epoch)
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))

    # Decoding all at once. Note that this isn't the best way

    # Padding input to max_time_step of this batch
    indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

    batch_train_inputs = train_inputs[indexes]
            # Padding input to max_time_step of this batch
    batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
            # Converting to sparse representation so as to to feed SparseTensor input
    batch_train_targets = sparse_tuple_from(train_targets[indexes])


    feed = {X: batch_train_inputs,
            Y: batch_train_targets,
            seq_len: batch_train_seq_len,
            keep_prob:0.5
            }

    # Decoding
    d = session.run(decoded[0], feed_dict=feed)
    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

    for i, seq in enumerate(dense_decoded):

        seq = [s for s in seq if s != -1]

        print('Sequence %d' % i)
        print('\t Original:\n%s' % train_targets[i])
        print('\t Decoded:\n%s' % seq)
