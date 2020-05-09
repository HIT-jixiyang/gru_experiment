import tensorflow as tf
import numpy as np
import config as c
from generator import generator
from evaluation import Evaluator
from iterator import Iterator
import os
from utils import *
import logging
from tf_utils import *


def began_train(paras, mode='train', test_path='test', start_iter=0):
    config_log()
    batch_size = c.BATCH_SIZE
    graph = tf.Graph()
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    with graph.as_default():
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        with tf.device("/gpu:1"):
            real_in = tf.placeholder(tf.float32,
                                     [c.BATCH_SIZE, c.IN_SEQ, c.H, c.W, c.IN_CHANEL])
            real_pred = tf.placeholder(tf.float32,
                                       [c.BATCH_SIZE, c.OUT_SEQ, c.H, c.W, 1])
            real_hist=tf.placeholder(tf.float32,
                                     [13])
            gen_pred = generator(real_in, real_pred, c.BATCH_SIZE, c.IN_SEQ, c.OUT_SEQ, c.H, c.W, c.IN_CHANEL)
            grad_loss_1=sobel_loss(gen_pred,real_pred)
            grad_loss_2=lap_loss(gen_pred,real_pred)

            print('num',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            # tf.reduce_max
            # MAX_LOSS=tf.abs(tf.reduce_max(gen_pred[:,0])-tf.reduce_max(gen_pred[:,-1]))
            print('genpred',gen_pred)
            gen_pred_tra=tf.reshape(tf.transpose(gen_pred,[1,0,2,3,4]),(c.OUT_SEQ,-1))
            real_pred_tra=tf.reshape(tf.transpose(real_pred,[1,0,2,3,4]),(c.OUT_SEQ,-1))

            max_ABS=tf.abs(tf.reduce_max(gen_pred_tra, reduction_indices=[1]) - tf.reduce_max(real_pred_tra, reduction_indices=[1]))
            print('shape abs',max_ABS)
            MAX_LOSS=tf.reduce_mean(max_ABS)
            G_MAE = tf.reduce_mean(tf.abs(gen_pred - real_pred))
            # G_MAE1=tf.reduce_mean(tf.abs(gen_pred[:,:5]-real_pred[:,:5]))
            # weight1 = get_loss_weight_symbol(gen_pred)
            # G_MAE= weighted_mae(gen_pred, real_pred, weight1)
            G_MSE = tf.reduce_mean(tf.square(gen_pred - real_pred))
            G_GDL = gdl_loss(gen_pred, real_pred)
            SSIM=tf_ssim(gen_pred,real_pred)
            Hinge_loss=hinge_loss(gen_pred,real_pred,real_hist)
            LOSS=c.lam_hinge*Hinge_loss+c.lam_grad1*grad_loss_1+c.lam_grad2*grad_loss_2+c.lam_l1*G_MAE+c.lam_l2*G_MSE+c.lam_ssim*SSIM+c.lam_max*MAX_LOSS
        with tf.device("/gpu:0"):
            # loss=G_MAE+0.000001*G_GDL
            opt= tf.train.AdamOptimizer(c.GEN_LR)
            opt_gen = opt.minimize(LOSS)
            # with tf.control_dependencies(update_ops):
            #     opt_gen2=tf.train.AdamOptimizer(c.GEN_LR).minimize(G_MAE2)
        tf.summary.scalar('gen_mse', G_MSE)
        tf.summary.scalar('gen_mae', G_MAE)
        tf.summary.scalar('gen_gdl', G_GDL)
        tf.summary.scalar('ssim', SSIM)
        tf.summary.scalar('grad_1', grad_loss_1)
        tf.summary.scalar('grad_2', grad_loss_2)
        tf.summary.scalar('max_loss', MAX_LOSS)
        tf.summary.scalar('hinge_loss', Hinge_loss)
        summary_merge = tf.summary.merge_all()
        params = tf.trainable_variables()
        tr_vars = {}

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=0)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph,
                          config=tf_config)
        summary_writer = tf.summary.FileWriter(c.SAVE_SUMMARY, graph=sess.graph)
        if paras is not None:
            saver.restore(sess, paras)
        else:
            sess.run(init)
        print('start pred')
        iterator = Iterator(c.BATCH_SIZE)
        if mode == 'valid':
            batch_num = 1000 // batch_size
            for j in range(batch_num):
                valid_data, indexes = iterator.get_valid_batch()
                print(np.shape(valid_data))
                try:
                    valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])

                    # full_data[:,:]=data[:,:-1]
                    real_in_data = valid_data[:, :c.IN_SEQ, :]
                    real_pred_data = valid_data[:, c.IN_SEQ:, :]
                    pred, mae, mse,h_loss = sess.run([gen_pred, G_MAE, G_MSE,Hinge_loss],
                                              {real_in: real_in_data, real_pred: real_pred_data})
                    for b in range(batch_size):
                        save_pred(valid_data[b], pred[b], indexes[b],
                                  os.path.join(c.VALID_PATH, str(start_iter) + 'valid'))
                    logging.info("valid step{} mae {} mse {}".format(j, mae, mse))
                    logging.info("pred max{} min {}".format(np.max(pred), np.min(pred)))
                except Exception as e:
                    print(e)
                    continue
        if mode == 'test':
            for t in c.TEST_TIME:
                iterator = Iterator(c.BATCH_SIZE,test_time=t)

                test_data, dates = iterator.get_test_batch()
                # print(test_data.shape)
                # full_data[:,:]=data[:,:-1]
                # real_in_data = test_data[:, 0:c.IN_SEQ, :]
                # real_pred_data = test_data[:, c.IN_SEQ:, ]
                while test_data is not None:
                    try:
                        print(test_data.shape)
                        # full_data[:,:]=data[:,:-1]
                        real_in_data = test_data[:, 0:c.IN_SEQ, :]
                        real_pred_data = test_data[:, c.IN_SEQ:, ]
                        hist = getHist(real_pred_data)
                        # valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                        pred, mae, mse ,h_loss= sess.run([gen_pred, G_MAE, G_MSE,Hinge_loss],
                                                  {real_in: real_in_data, real_pred: real_pred_data,real_hist:hist})
                        for b in range(batch_size):
                            save_pred(test_data[b], pred[b, :, :, :, :1], dates[b][4],
                                      os.path.join(c.TEST_PATH, test_path + 'test'), color=True)

                        logging.info(" mae {} mse {}".format(mae, mse))
                        logging.info(
                            "pred max{} min {}".format(np.max(pred), np.min(pred)))
                        test_data, dates = iterator.get_test_batch()

                    except Exception as e:
                        print(e)
                        continue
        else:

            for iter in range(start_iter, c.MAX_ITER):

                data = iterator.get_batch()
                if data is None:
                    continue
                print(data.shape)
                data = np.reshape(data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                real_in_data = data[:, 0:c.IN_SEQ, :]
                real_pred_data = data[:, c.IN_SEQ:]
                hist=getHist(real_pred_data)
                print("---------------------------------{}--------------------------------".format(iter))

                g_pred, g_mae, g_mse,grad_1,grad_2, max_loss,h_loss,ssim,summary, _ = \
                    sess.run([gen_pred, G_MAE, G_MSE,grad_loss_1,grad_loss_2,MAX_LOSS,Hinge_loss,SSIM,summary_merge, opt_gen],
                             {
                                 real_in: real_in_data,
                                 real_pred: real_pred_data,
                                 real_hist:hist
                             }
                             )
                print(iter, 'g_mae', g_mae, 'g_mse', g_mse,'grad_1',grad_1,'grad_2',grad_2,'max_loss',max_loss,'ssim',ssim,'h_loss',h_loss)
                for st in range(c.OUT_SEQ):
                    print(np.max(real_pred_data[:, st]), end=' ')
                print('\n')
                for st in range(c.OUT_SEQ):
                    print(np.max(g_pred[:, st]), end=' ')
                print('\n ')
                if (iter + 1) % c.SUMMARY_ITER == 0:
                    summary_writer.add_summary(summary, iter)
                if (iter + 1) % c.SAVE_ITER == 0:
                    saver.save(sess, os.path.join(c.SAVE_PATH, "model.ckpt"),
                               global_step=iter)
                if (iter + 1) % c.VALID_ITER == 0:
                    batch_num = c.VALID__SEQ // batch_size
                    evaluator = Evaluator(os.path.join(c.SAVE_METRIC, str(iter)), seq=c.OUT_SEQ)
                    for j in range(batch_num):
                        valid_data, indexes = iterator.get_valid_batch()
                        try:
                            valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                            # full_data[:,:]=data[:,:-1]
                            real_in_data = valid_data[:, :c.IN_SEQ, :]
                            real_pred_data = valid_data[:, c.IN_SEQ:, :]
                            hist=getHist(real_pred_data)
                            pred, mae, mse, grad_1, grad_2, max_loss, ssim,h_loss = \
                                sess.run(
                                    [gen_pred, G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS, SSIM,Hinge_loss],
                                    {
                                        real_in: real_in_data,
                                        real_pred: real_pred_data,
                                        real_hist:hist
                                    }
                                    )

                            evaluator.evaluate(real_pred_data, pred)
                            for b in range(batch_size):
                                save_pred(valid_data[b], pred[b], indexes[b],
                                          os.path.join(c.VALID_PATH, str(iter) + 'valid'))
                            logging.info("valid step{} mae {} mse {}".format(j, mae, mse))
                            logging.info("pred max{} min {}".format(np.max(pred), np.min(pred)))
                        except:
                            continue
                    evaluator.done()
# def test()

if __name__ == '__main__':
    # began_train(None,start_iter=0)
    os.environ['CUDA_VISIBLE_DEVICES']='3,4'
    # began_train('/extend/gru_tf_data/gru_tf_grad/Save/model.ckpt-199999',start_iter=120000,mode='test',test_path='199999-qpe-345678')
    began_train(paras=None,start_iter=0)
    # began_train(paras=N
    # ite = Iterator(2)
    # data = ite.get_batch()
    # print(getHist(data))
    # test()
    # tf.contrib.layers.layer_norm