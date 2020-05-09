import tensorflow as tf
import numpy as np
import config as c
from generator import generator
from evaluation import Evaluator
from iterator import Iterator
import os
from utils import *
import logging
# test()
iterator=Iterator(1)
test_data, dates = iterator.get_test_batch()
print(test_data.shape)
# full_data[:,:]=data[:,:-1]
real_in_data = test_data[:, 0:c.IN_SEQ, :].astype(np.float32)
real_pred_data = test_data[:,1, ].astype(np.float32)
vars_full=np.var(real_pred_data,axis=(1,2,3),keepdims=True)
mean_full=np.mean(real_pred_data,axis=(1,2,3),keepdims=True)
vars_360=np.var(real_pred_data[:,300:660,300:660],axis=(1,2,3),keepdims=True)
print('max of full',np.max(real_pred_data))
print('max of 360',np.max(real_pred_data[:,300:660,300:660]))
mean_360=np.mean(real_pred_data[:,300:660,300:660],axis=(1,2,3),keepdims=True)
print('vars of full',vars_full,'vars of 360',vars_360)
print('mean of full',mean_full,'mean of 360',mean_360)
x_normalized_full=(real_pred_data-mean_full)/np.sqrt(vars_full+1e-6)
x_normalized_360=(real_pred_data[:,300:660,300:660]-mean_360)/np.sqrt(vars_360+1e-6)

print('max of full after norm',np.max(2*x_normalized_full+0.5))
print('max of 360 after norm',np.max(2*x_normalized_360+0.5))
