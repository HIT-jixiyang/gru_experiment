from evaluation import Evaluator
#from generator import generator
from generator import generator
from iterator import Iterator
from tf_utils import *
from utils import *
import traceback
import cv2
def get_index(array):
    m=np.max(array)
    for i in range(len(array)):
        if array[i]==m:
            return i
    return 0


def Guass_blur(data):
    """

    :param data: batch*N*W*H
    :return:
    """
    shape=data.shape
    if len(shape)!=3:
        data=np.reshape(data,[shape[0]*shape[1],shape[2],shape[3]])
    res=np.zeros_like(data)
    for i in range(len(data)):
        res[i]=cv2.GaussianBlur(data[i],ksize=(3,3),sigmaX=1)
    if len(shape)!=3:
        return np.reshape(res,[shape[0],shape[1],shape[2],shape[3]])
    else:
        return res

def reg_classfi_model(paras, mode='train', test_path='test', start_iter=0):
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
            real_hist = tf.placeholder(tf.float32,
                                       [13])
            real_logits = tf.placeholder(tf.float32,
                                         [c.BATCH_SIZE, c.OUT_SEQ, c.H, c.W, 17])

            gen_pred, pred_logit, en_f, de_f = generator(real_in, real_pred, c.BATCH_SIZE, c.IN_SEQ, c.OUT_SEQ, c.H,
                                                         c.W,
                                                         c.IN_CHANEL)
            pred_logit=tf.reshape(pred_logit,[batch_size*c.OUT_SEQ*c.H*c.W,17])
            real_logits=tf.reshape(real_logits,[batch_size*c.OUT_SEQ*c.H*c.W,17])

            CL_LOSS=multi_category_focal_loss1(real_logits,pred_logit)
            if c.lam_grad1>0:
                grad_loss_1 = sobel_loss(gen_pred, real_pred)
            else:
                grad_loss_1=tf.reduce_mean(tf.zeros([1]))

            grad_loss_2 = lap_loss(gen_pred, real_pred)
            print('num', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            # tf.reduce_max
            # MAX_LOSS=tf.abs(tf.reduce_max(gen_pred[:,0])-tf.reduce_max(gen_pred[:,-1]))
            print('genpred', gen_pred)
            gen_pred_tra = tf.reshape(tf.transpose(gen_pred, [1, 0, 2, 3, 4]), (c.OUT_SEQ, -1))
            real_pred_tra = tf.reshape(tf.transpose(real_pred, [1, 0, 2, 3, 4]), (c.OUT_SEQ, -1))

            max_ABS = tf.abs(tf.reduce_max(gen_pred_tra, reduction_indices=[1]) - tf.reduce_max(real_pred_tra,
                                                                                                reduction_indices=[1]))
            print('shape abs', max_ABS)
            if c.lam_max>0:
                MAX_LOSS = max_pool_loss(gen_pred,real_pred)
            else:
                MAX_LOSS =tf.reduce_mean(tf.zeros([1]))
            G_MAE = tf.reduce_mean(tf.abs(gen_pred - real_pred))
            G_MSE = tf.reduce_mean(tf.square(gen_pred - real_pred))
            if c.lam_gdl>0:
                G_GDL = gdl_loss(gen_pred, real_pred)
            else:
                G_GDL=tf.reduce_mean(tf.zeros([1]))
            if c.lam_ssim>0:
                SSIM = tf_multi_ssim(gen_pred, real_pred)
            else:
                SSIM= tf.reduce_mean(tf.zeros([1]))
            # Hinge_loss = hinge_loss(gen_pred, real_pred, real_hist)
            LOSS =  CL_LOSS*c.lam_cl+c.lam_grad1 * grad_loss_1 + c.lam_grad2 * grad_loss_2 + c.lam_l1 * G_MAE + c.lam_l2 * G_MSE + c.lam_ssim * SSIM + c.lam_max * MAX_LOSS
        with tf.device("/gpu:0"):
            loss = G_MAE + 0.000001 * G_GDL
            opt = tf.train.AdamOptimizer(c.GEN_LR)
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
        tf.summary.scalar('focalloss', CL_LOSS)
        summary_merge = tf.summary.merge_all()
        params = tf.trainable_variables()
        tr_vars = {}
        print(params)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=0)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction =0.5
        sess = tf.Session(graph=graph,
                          config=tf_config)
        summary_writer = tf.summary.FileWriter(c.SAVE_SUMMARY, graph=sess.graph)
        if paras is not None:
            sess.run(init)
            saver.restore(sess, paras)
        else:
            sess.run(init)
        print('start pred')
        iterator = Iterator(c.BATCH_SIZE)
        if mode == 'valid':
            batch_num = c.VALID__SEQ // batch_size
            evaluator = Evaluator(os.path.join(c.SAVE_METRIC, str(start_iter)), seq=c.OUT_SEQ)
            evaluator_cl = Evaluator(os.path.join(c.SAVE_METRIC, str(start_iter) + 'cl'), seq=c.OUT_SEQ)
            valid_log = open(os.path.join(c.VALID_PATH, 'valid.log'), 'a')
            mae_total = 0
            mse_total = 0
            cl_loss_total = 0
            for j in range(batch_num):
                valid_data, indexes = iterator.get_valid_batch()
                try:
                    valid_data = Guass_blur(valid_data)

                    data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])

                    real_in_data = data[:, 0:c.IN_SEQ, :]
                    # real_pred_data = data[:, c.IN_SEQ:]
                    th_data = (data + 2) // 5
                    th_real_in_data = th_data[:, 0:c.IN_SEQ, :]
                    th_real_in_data = th_real_in_data * 5
                    th_real_pred_data = th_data[:, c.IN_SEQ:]
                    th_real_pred_data = np.reshape(th_real_pred_data, [batch_size * c.OUT_SEQ * c.H * c.W])

                    logit = np.zeros([batch_size * c.OUT_SEQ * c.H * c.W, 17])
                    for i in range(len(data)):
                        logit[i, th_real_pred_data[i]] = 1
                    logit = np.reshape(logit, [batch_size * c.OUT_SEQ * c.H * c.W, 17])

                    real_in_data = np.concatenate((real_in_data, th_real_in_data), axis=-1)

                    real_pred_data = data[:, c.IN_SEQ:]

                    pred, _pred_logit, cl_loss, mae, mse, grad_1, grad_2, max_loss, ssim = \
                        sess.run(
                            [gen_pred, pred_logit, CL_LOSS, G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS,
                             SSIM],
                            {
                                real_in: real_in_data,
                                real_pred: real_pred_data,
                                real_logits: logit
                            }
                        )
                    focal_pred = np.zeros([batch_size * c.OUT_SEQ * c.H * c.W, 1])

                    for index in range(len(_pred_logit)):
                        focal_pred[index] = get_index(_pred_logit[index]) * 5

                    focal_pred = np.reshape(focal_pred, [batch_size, c.OUT_SEQ, c.H, c.W, 1])
                    _pred_logit = np.reshape(_pred_logit, [batch_size, c.OUT_SEQ, c.H, c.W, 17])

                    evaluator.evaluate(real_pred_data, pred)
                    evaluator_cl.evaluate(real_pred_data, pred)
                    if j % 10 == 0:
                        for b in range(batch_size):
                            save_pred(data[b], pred[b], indexes[b],
                                      os.path.join(c.VALID_PATH, str(start_iter) + 'valid'))
                        for b in range(batch_size):
                            save_pred(data[b], focal_pred[b], indexes[b],
                                      os.path.join(c.VALID_PATH, str(start_iter) + 'focalvalid'))
                    logging.info("valid step{} mae {} mse {}".format(j, mae, mse))

                    logging.info("pred max{} min {}".format(np.max(pred), np.min(pred)))
                    mae_total += mae
                    mse_total += mse
                    cl_loss_total += cl_loss

                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    continue
            mae_total = mae_total / batch_num
            mse_total = mse_total / batch_num
            cl_loss_total = cl_loss_total / batch_num
            valid_log.write('mae-{}  mse-{} cl_loss-{}'.format(mae_total, mse_total, cl_loss_total))
            valid_log.flush()
            evaluator.done()
            evaluator_cl.done()
        if mode == 'test':
            for t in c.TEST_TIME:
                iterator = Iterator(c.BATCH_SIZE, test_time=t)

                test_data, dates = iterator.get_test_batch()
                # print(test_data.shape)
                # full_data[:,:]=data[:,:-1]
                # real_in_data = test_data[:, 0:c.IN_SEQ, :]
                # real_pred_data = test_data[:, c.IN_SEQ:, ]

                while test_data is not None:
                    try:
                        print(test_data.shape)
                        print('max of data', np.max(test_data))
                        # full_data[:,:]=data[:,:-1]
                        real_in_data = np.zeros([c.BATCH_SIZE, c.IN_SEQ, c.H, c.W, c.IN_CHANEL])
                        real_pred_data = np.zeros([c.BATCH_SIZE, c.OUT_SEQ, c.H, c.W, c.IN_CHANEL])
                        if -(c.H - 912) // 2 != 0:

                            real_in_data[:, :, (c.H - 912) // 2:-(c.H - 912) // 2,
                            (c.H - 912) // 2:-(c.H - 912) // 2] = test_data[:, 0:c.IN_SEQ]
                            real_pred_data[:, :, (c.H - 912) // 2:-(c.H - 912) // 2,
                            (c.H - 912) // 2:-(c.H - 912) // 2] = test_data[:, c.IN_SEQ:]
                        else:
                            real_in_data[:, :, :, :] = test_data[:, 0:c.IN_SEQ]
                            real_pred_data[:, :, :, :] = test_data[:, c.IN_SEQ:]

                        # real_in_data = test_data[:, 0:c.IN_SEQ]
                        # real_pred_data = test_data[:, c.IN_SEQ:]

                        hist = getHist(real_pred_data)
                        # valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                        pred, mae, mse, h_loss, e_f, d_f = sess.run([gen_pred, G_MAE, G_MSE, en_f, de_f],
                                                                    {real_in: real_in_data, real_pred: real_pred_data,
                                                                     real_hist: hist})
                        for b in range(batch_size):
                            save_pred(
                                test_data[b, :, (912 - 700) // 2:-(912 - 700) // 2, (912 - 900) // 2:-(912 - 900) // 2],
                                pred[b, :, (c.H - 700) // 2:-(c.H - 700) // 2, (c.W - 900) // 2:-(c.W - 900) // 2, :1],
                                dates[b][4],
                                os.path.join(c.TEST_PATH, test_path + 'test'), color=True)
                            # np.save('./en_f_{}.npy'.format(dates[b][4]), e_f)
                            # np.save('./de_f_{}.npy'.format(dates[b][4]), d_f)
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
                th_data = (Guass_blur(data) + 2) // 5
                th_real_in_data = np.reshape(th_data[:, 0:c.IN_SEQ, :],[batch_size, c.IN_SEQ , c.H, c.W, 1])
                th_real_in_data = th_real_in_data * 5
                th_real_pred_data = th_data[:, c.IN_SEQ:]
                th_real_pred_data = np.reshape(th_real_pred_data, [batch_size * c.OUT_SEQ * c.H * c.W])
                data = np.reshape(data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                real_in_data = data[:, 0:c.IN_SEQ, :]
                real_pred_data = data[:, c.IN_SEQ:]
                # logit = np.zeros([batch_size * c.OUT_SEQ * c.H * c.W, 17])
                n_values = 17
                logit=np.eye(n_values)[th_real_pred_data]


                hist = getHist(real_pred_data)
                real_in_data = np.concatenate((real_in_data, th_real_in_data), axis=-1)

                real_pred_data = data[:, c.IN_SEQ:]

                print("---------------------------------{}--------------------------------".format(iter))

                g_pred, cl_loss,g_mae, g_mse, grad_1, grad_2, max_loss, ssim, summary, _ =sess.run(
                    [gen_pred, CL_LOSS,G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS, SSIM, summary_merge,opt_gen],{real_in: real_in_data,
                    real_pred: real_pred_data,
                    real_hist: hist,
                    real_logits: logit
                }
                )
                print(iter, 'cl_loss',cl_loss,'g_mae', g_mae, 'g_mse', g_mse, 'grad_1', grad_1, 'grad_2', grad_2, 'max_loss', max_loss,
                      'ssim', ssim)
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
                    evaluator_cl = Evaluator(os.path.join(c.SAVE_METRIC, str(iter) + 'cl'), seq=c.OUT_SEQ)
                    valid_log = open(os.path.join(c.VALID_PATH, 'valid.log'), 'a')
                    mae_total = 0
                    mse_total = 0
                    cl_loss_total = 0
                    for j in range(batch_num):
                        valid_data, indexes = iterator.get_valid_batch()
                        try:
                            valid_data = Guass_blur(valid_data)

                            data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])

                            real_in_data = data[:, 0:c.IN_SEQ, :]
                            # real_pred_data = data[:, c.IN_SEQ:]
                            th_data = (data + 2) // 5
                            th_real_in_data = th_data[:, 0:c.IN_SEQ, :]
                            th_real_in_data = th_real_in_data * 5
                            th_real_pred_data = th_data[:, c.IN_SEQ:]
                            th_real_pred_data = np.reshape(th_real_pred_data, [batch_size * c.OUT_SEQ * c.H * c.W])

                            n_values = 17
                            logit = np.eye(n_values)[th_real_pred_data]
                            real_in_data = np.concatenate((real_in_data, th_real_in_data), axis=-1)

                            real_pred_data = data[:, c.IN_SEQ:]

                            pred, _pred_logit, cl_loss, mae, mse, grad_1, grad_2, max_loss, ssim = \
                                sess.run(
                                    [gen_pred, pred_logit, CL_LOSS, G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS,
                                     SSIM],
                                    {
                                        real_in: real_in_data,
                                        real_pred: real_pred_data,
                                        real_logits: logit
                                    }
                                )
                            focal_pred = np.zeros([batch_size * c.OUT_SEQ * c.H * c.W, 1])

                            for index in range(len(_pred_logit)):
                                focal_pred[index] = get_index(_pred_logit[index]) * 5

                            focal_pred = np.reshape(focal_pred, [batch_size, c.OUT_SEQ, c.H, c.W, 1])
                            _pred_logit = np.reshape(_pred_logit, [batch_size, c.OUT_SEQ, c.H, c.W, 17])

                            evaluator.evaluate(real_pred_data, pred)
                            evaluator_cl.evaluate(real_pred_data, pred)
                            if j % 10 == 0:
                                for b in range(batch_size):
                                    save_pred(data[b], pred[b], indexes[b],
                                              os.path.join(c.VALID_PATH, str(iter) + 'valid'))
                                for b in range(batch_size):
                                    save_pred(data[b], focal_pred[b], indexes[b],
                                              os.path.join(c.VALID_PATH, str(iter) + 'focalvalid'))
                            logging.info("valid step{} mae {} mse {}".format(j, mae, mse))

                            logging.info("pred max{} min {}".format(np.max(pred), np.min(pred)))
                            mae_total += mae
                            mse_total += mse
                            cl_loss_total += cl_loss

                        except:
                            continue
                    mae_total = mae_total / batch_num
                    mse_total = mse_total / batch_num
                    cl_loss_total = cl_loss_total / batch_num
                    valid_log.write('mae-{}  mse-{} cl_loss-{}'.format(mae_total, mse_total, cl_loss_total))
                    valid_log.flush()
                    evaluator.done()
                    evaluator_cl.done()


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
            real_hist = tf.placeholder(tf.float32,
                                       [13])
            gen_pred, _, en_f, de_f = generator(real_in, real_pred, c.BATCH_SIZE, c.IN_SEQ, c.OUT_SEQ, c.H, c.W,
                                                c.IN_CHANEL,incept=False)
            grad_loss_1 = sobel_loss(gen_pred, real_pred)
            grad_loss_2 = lap_loss(gen_pred, real_pred)

            print('num', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            # tf.reduce_max
            # MAX_LOSS=tf.abs(tf.reduce_max(gen_pred[:,0])-tf.reduce_max(gen_pred[:,-1]))
            print('genpred', gen_pred)
            gen_pred_tra = tf.reshape(tf.transpose(gen_pred, [1, 0, 2, 3, 4]), (c.OUT_SEQ, -1))
            real_pred_tra = tf.reshape(tf.transpose(real_pred, [1, 0, 2, 3, 4]), (c.OUT_SEQ, -1))

            max_ABS = tf.abs(tf.reduce_max(gen_pred_tra, reduction_indices=[1]) - tf.reduce_max(real_pred_tra,
                                                                                                reduction_indices=[1]))
            print('shape abs', max_ABS)
            MAX_LOSS = max_pool_loss(gen_pred,real_pred)
            abs = tf.abs(gen_pred - real_pred)
            abs = tf.transpose(abs, [0, 4, 2, 3, 1]) * [1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4, 1.4, 1.6, 1.6,
                                                        1.6, 1.6, 1.6, 1.6, 1.6, 1.6]

            print('abs', abs)
            G_MAE = tf.reduce_mean(abs)
            # G_MAE = tf.reduce_mean(tf.abs(gen_pred - real_pred))
            # G_MAE1=tf.reduce_mean(tf.abs(gen_pred[:,:5]-real_pred[:,:5]))
            # weight1 = get_loss_weight_symbol(gen_pred)
            # G_MAE= weighted_mae(gen_pred, real_pred, weight1)
            G_MSE = tf.reduce_mean(tf.square(gen_pred - real_pred))
            G_GDL = gdl_loss(gen_pred, real_pred)
            SSIM = tf_multi_ssim(gen_pred, real_pred)
            Hinge_loss = hinge_loss(gen_pred, real_pred, real_hist)
            LOSS = c.lam_hinge * Hinge_loss + c.lam_grad1 * grad_loss_1 + c.lam_grad2 * grad_loss_2 + c.lam_l1 * G_MAE + c.lam_l2 * G_MSE + c.lam_ssim * SSIM + c.lam_max * MAX_LOSS
        with tf.device("/gpu:0"):
            loss = G_MAE + 0.000001 * G_GDL
            opt = tf.train.AdamOptimizer(c.GEN_LR)
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
        print(params)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=0)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph,
                          config=tf_config)
        summary_writer = tf.summary.FileWriter(c.SAVE_SUMMARY, graph=sess.graph)
        if paras is not None:
            sess.run(init)
            saver.restore(sess, paras)
        else:
            sess.run(init)
        print('start pred')
        iterator = Iterator(c.BATCH_SIZE)
        if mode == 'valid':
            batch_num = c.VALID__SEQ // batch_size
            evaluator = Evaluator(os.path.join(c.SAVE_METRIC, str(start_iter)), seq=c.OUT_SEQ)
            for j in range(batch_num):
                print('--------------------{}-----------------'.format(j))
                valid_data, indexes = iterator.get_valid_batch()
                try:
                    valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                    # full_data[:,:]=data[:,:-1]
                    real_in_data = valid_data[:, :c.IN_SEQ, :]
                    real_pred_data = valid_data[:, c.IN_SEQ:, :]
                    hist = getHist(real_pred_data)
                    pred, mae, mse, grad_1, grad_2, max_loss, ssim, h_loss = \
                        sess.run(
                            [gen_pred, G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS, SSIM, Hinge_loss],
                            {
                                real_in: real_in_data,
                                real_pred: real_pred_data,
                                real_hist: hist
                            }
                        )

                    evaluator.evaluate(real_pred_data, pred)
                    for b in range(batch_size):
                        save_pred(valid_data[b], pred[b], indexes[b],
                                  os.path.join(c.VALID_PATH, str(start_iter) + 'valid'))
                    logging.info("valid step{} mae {} mse {}".format(j, mae, mse))
                    logging.info("pred max{} min {}".format(np.max(pred), np.min(pred)))
                except:
                    continue
            evaluator.done()
        if mode == 'eval':
            iterator = Iterator(c.BATCH_SIZE, test_time=c.TEST_TIME[0])

            test_data, dates = iterator.get_eval_batch()
            # print(test_data.shape)
            # full_data[:,:]=data[:,:-1]
            # real_in_data = test_data[:, 0:c.IN_SEQ, :]
            # real_pred_data = test_data[:, c.IN_SEQ:, ]
            while test_data is not None:
                try:
                    print(test_data.shape)
                    print('max of data', np.max(test_data))
                    # full_data[:,:]=data[:,:-1]
                    real_in_data = test_data[:, 0:c.IN_SEQ]
                    real_pred_data = test_data[:, c.IN_SEQ:]
                    hist = getHist(real_pred_data)
                    # valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                    pred, mae, mse, h_loss, e_f, d_f = sess.run([gen_pred, G_MAE, G_MSE, Hinge_loss, en_f, de_f],
                                                                {real_in: real_in_data, real_pred: real_pred_data,
                                                                 real_hist: hist})
                    for b in range(batch_size):
                        save_pred(test_data[b, :], pred[b, :, :, :, :1], dates[b][4],
                                  os.path.join(c.TEST_PATH, test_path + 'eval'), color=False)

                    logging.info(" mae {} mse {}".format(mae, mse))
                    logging.info(
                        "pred max{} min {}".format(np.max(pred), np.min(pred)))
                    test_data, dates = iterator.get_eval_batch()

                except Exception as e:
                    print(e)
                    continue
        if mode == 'test':
            for t in c.TEST_TIME:
                iterator = Iterator(c.BATCH_SIZE, test_time=t)

                test_data, dates = iterator.get_test_batch()
                # print(test_data.shape)
                # full_data[:,:]=data[:,:-1]
                # real_in_data = test_data[:, 0:c.IN_SEQ, :]
                # real_pred_data = test_data[:, c.IN_SEQ:, ]

                while test_data is not None:
                    try:
                        print(test_data.shape)
                        print('max of data', np.max(test_data))
                        # full_data[:,:]=data[:,:-1]
                        real_in_data = np.zeros([c.BATCH_SIZE, c.IN_SEQ, c.H, c.W, c.IN_CHANEL])
                        real_pred_data = np.zeros([c.BATCH_SIZE, c.OUT_SEQ, c.H, c.W, c.IN_CHANEL])
                        if -(c.H - 912) // 2 != 0:

                            real_in_data[:, :, (c.H - 912) // 2:-(c.H - 912) // 2,
                            (c.H - 912) // 2:-(c.H - 912) // 2] = test_data[:, 0:c.IN_SEQ]
                            real_pred_data[:, :, (c.H - 912) // 2:-(c.H - 912) // 2,
                            (c.H - 912) // 2:-(c.H - 912) // 2] = test_data[:, c.IN_SEQ:]
                        else:
                            real_in_data[:, :, :, :] = test_data[:, 0:c.IN_SEQ]
                            real_pred_data[:, :, :, :] = test_data[:, c.IN_SEQ:]

                        # real_in_data = test_data[:, 0:c.IN_SEQ]
                        # real_pred_data = test_data[:, c.IN_SEQ:]

                        hist = getHist(real_pred_data)
                        # valid_data = np.reshape(valid_data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                        pred, mae, mse, h_loss, e_f, d_f = sess.run([gen_pred, G_MAE, G_MSE, Hinge_loss, en_f, de_f],
                                                                    {real_in: real_in_data, real_pred: real_pred_data,
                                                                     real_hist: hist})
                        for b in range(batch_size):
                            save_pred(
                                test_data[b, :, (912 - 700) // 2:-(912 - 700) // 2, (912 - 900) // 2:-(912 - 900) // 2],
                                pred[b, :, (c.H - 700) // 2:-(c.H - 700) // 2, (c.W - 900) // 2:-(c.W - 900) // 2, :1],
                                dates[b][4],
                                os.path.join(c.TEST_PATH, test_path + 'test'), color=True)
                            # np.save('./en_f_{}.npy'.format(dates[b][4]), e_f)
                            # np.save('./de_f_{}.npy'.format(dates[b][4]), d_f)
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
                print(iter,data.shape)
                data = np.reshape(data, [batch_size, c.IN_SEQ + c.OUT_SEQ, c.H, c.W, 1])
                real_in_data = data[:, 0:c.IN_SEQ, :]
                real_pred_data = data[:, c.IN_SEQ:]
                hist = getHist(real_pred_data)
                print("---------------------------------{}--------------------------------".format(iter))

                g_pred, g_mae, g_mse, grad_1, grad_2, max_loss, h_loss, ssim, summary, _ = \
                    sess.run(
                        [gen_pred, G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS, Hinge_loss, SSIM, summary_merge,
                         opt_gen],
                        {
                            real_in: real_in_data,
                            real_pred: real_pred_data,
                            real_hist: hist
                        }
                        )
                print(iter, 'g_mae', g_mae, 'g_mse', g_mse, 'grad_1', grad_1, 'grad_2', grad_2, 'max_loss', max_loss,
                      'ssim', ssim, 'h_loss', h_loss)
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
                            hist = getHist(real_pred_data)
                            pred, mae, mse, grad_1, grad_2, max_loss, ssim, h_loss = \
                                sess.run(
                                    [gen_pred, G_MAE, G_MSE, grad_loss_1, grad_loss_2, MAX_LOSS, SSIM, Hinge_loss],
                                    {
                                        real_in: real_in_data,
                                        real_pred: real_pred_data,
                                        real_hist: hist
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # began_train('/extend/gru_tf_data/gru_tf_grad/Save/model.ckpt-199999',start_iter=120000,mode='test',test_path='199999-qpe-345678')
    # began_train(paras='/extend/gru_tf_data/gru_experiment/instan_221_online_reuse/Save/model.ckpt-149999',start_iter=39999,mode='test',test_path='134999')
    reg_classfi_model(paras=None,mode='train',start_iter=0)
    began_train(paras=None,mode='train')
    # ite = Iterator(2)
    # data = ite.get_batch()
    # print(getHist(data))
    # test()
    # tf.contrib.layers.layer_norm
