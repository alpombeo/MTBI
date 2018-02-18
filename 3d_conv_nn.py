from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import io
import scipy
import os, glob
import pickle
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def mask_data(dir_name, mask_names, data):

    mask = np.zeros((1,182,218,182,1))
    os.chdir(dir_name)
    
    for mask_name in mask_names:
        
        mask_temp = scipy.io.loadmat(mask_name)
        mask_temp = mask_temp['vol']
        mask_temp = np.reshape(mask_temp, (1,182,218,182,1))
        mask = np.logical_or(mask_temp, mask)

           

    data = data*mask

    return data


def load_data(folder_list, dir_name, dim = [-1,182,218,182,1]):
    
    labels_new= np.hstack( ( np.ones((28,)), np.zeros((21,)) ) )
    labels_old= np.hstack( ( np.ones((42,)), np.zeros((23,)) ) )
    labels = np.hstack( ( labels_new, labels_old ) )
    fresh = 1
    i = 0
    for folder in folder_list:
        os.chdir(dir_name + folder)
        print('Starting on folder ' + folder)
        for file in glob.glob("*.mat"):
            if fresh == 1:
                data_temp = scipy.io.loadmat(file)
                data_temp2 = data_temp['vol']
                data = np.reshape(data_temp2, dim)
                data = np.asarray(data)
                fresh = 0
            else:
                data_temp = scipy.io.loadmat(file)
                data_temp2 = data_temp['vol']
                data_temp2 = np.reshape(data_temp2, dim)
                data_temp2 = np.asarray(data_temp2)
                data = np.concatenate([data, data_temp2], 0)
                i = i+1
                if i > 5:
                    return data, labels
                print(data.shape)


    return data, labels


def aec_make_model(input_data,
              batch_size = 1,
              kernel_size = [3,3,3], 
              strides = [1,2,2,2,1], 
              filters = [1,8,32,64],
              t_strides = [1,2,2,2,1],
              pool_size = [1,2,2,2,1],
              pool_strides = [1,1,1,1,1]):
    
    #128x218x182x1
#    input_layer = tf.reshape(input_data, input_dim)
#    input_layer = tf.cast(input_layer, tf.float32)
    
    #----------------- ENCODING ----------
    
    kernel1 = np.append(kernel_size, np.array([filters[0], filters[1]]))
    filters1 = tf.Variable(tf.ones(kernel1), dtype = tf.float32)
    
    #91x109x91x8
    conv1 = tf.nn.conv3d(input_data, filters1, strides, 'SAME')

    cae1 = tf.nn.max_pool3d(conv1, pool_size, pool_strides, 'SAME')

    kernel2 = np.append(kernel_size, np.array([filters[1], filters[2]]))
    filters2 = tf.Variable(tf.ones(kernel2), dtype = tf.float32)
    #46x55x46x32
    conv2 = tf.nn.conv3d(cae1, filters2, strides, 'SAME')

    cae2 = tf.nn.max_pool3d(conv2, pool_size, pool_strides, 'SAME')
    
    kernel3 = np.append(kernel_size, np.array([filters[2], filters[3]]))
    filters3 = tf.Variable(tf.ones(kernel3), dtype = tf.float32)
    #23x28x23x64
    conv3 = tf.nn.conv3d(cae2, filters3, strides, 'SAME')

    cae3 = tf.nn.max_pool3d(conv3, pool_size, pool_strides, 'SAME')

    

    #-----------------DECODING------------------
    output_shape1 = tf.constant([batch_size,46,55,46,32])
    t_kernel1 = np.append(kernel_size, np.array([filters[2], filters[3]]))
    t_filters1 = tf.Variable(tf.ones(t_kernel1), dtype = tf.float32)

    t_conv1 = tf.nn.conv3d_transpose(cae3, t_filters1, output_shape1,  t_strides)
    
    output_shape2 = tf.constant([batch_size,91,109,91,8])
    t_kernel2 = np.append(kernel_size, np.array([filters[1], filters[2]]))
    t_filters2 = tf.Variable(tf.ones(t_kernel2), dtype = tf.float32)
    
    t_conv2 = tf.nn.conv3d_transpose(t_conv1, t_filters2, output_shape2,  t_strides)
    
    output_shape3 = tf.constant([batch_size,182,218,182,1])
    t_kernel3 = np.append(kernel_size, np.array([filters[0], filters[1]])) 
    t_filters3 = tf.Variable(tf.ones(t_kernel3), dtype = tf.float32)
    t_conv3 = tf.nn.conv3d_transpose(t_conv2, t_filters3, output_shape3,  t_strides)
    
    return t_conv3



folder_list = ['AK', 'FA', 'MD', 'MK', 'RK']
dir_name = '/Users/alpombeo/Documents/Docs/NYU/RESEARCH/MTBI/'
full_data, labels = load_data(folder_list, dir_name)
mask_names = ['CC_Body_mask.mat', 'CC_Genu_mask.mat', 'CC_Splenium_mask.mat']

inputs_ = tf.placeholder(tf.float32, (1,182,218,182,1), name="input")
targets_ = tf.placeholder(tf.float32, (1,182,218,182,1), name="target")

lr = 0.001

masked = mask_data(dir_name, mask_names, full_data)

aec_out = aec_make_model(inputs_)

aec_loss = tf.reduce_mean(tf.square(inputs_ - aec_out))

opt = tf.train.AdamOptimizer(lr).minimize(aec_loss)


###TRAINING

sess = tf.Session()

epochs = 10
batch_size = 1
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for ii in range(full_data.shape[0]//batch_size):
        imgs = masked[ii:ii+batch_size:1,:,:,:,:]
        batch_cost, _ = sess.run([opt, aec_loss], feed_dict={inputs_: imgs,
                                                         targets_: imgs})

        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))







