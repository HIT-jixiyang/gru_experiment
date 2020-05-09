import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import config as c
from multiprocessing import Pool
import PIL.Image as Img
from concurrent.futures import ThreadPoolExecutor, wait
from map_of_color import color_map
import scipy.misc
import sys
import cv2
color = {
    0: [0, 0, 0, 0],
    1: [0, 236, 236, 255],
    2: [1, 160, 246, 255],
    3: [1, 0, 246, 255],
    4: [0, 239, 0, 255],
    5: [0, 200, 0, 255],
    6: [0, 144, 0, 255],
    7: [255, 255, 0, 255],
    8: [231, 192, 0, 255],
    9: [255, 144, 2, 255],
    10: [255, 0, 0, 255],
    11: [166, 0, 0, 255],
    12: [101, 0, 0, 255],
    13: [255, 0, 255, 255],
    14: [153, 85, 201, 255],
    15: [255, 255, 255, 255],
    16: [0, 0, 0, 0]
}

gray_cursor = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 1000]
_imread_executor_pool = ThreadPoolExecutor(max_workers=16)

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def laplacian(data):
    # 在某个维度上算梯度
    gauss_op = tf.constant(
        np.reshape(np.array([[0,0,1,0,0], [0,1,2,1,0], [1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]]),
                   [5, 5, 1, 1]), dtype=tf.float32)
    input_size = data.shape.as_list()
    if len(input_size) == 5:
        data = tf.reshape(data, shape=(input_size[0] * input_size[1],
                                       input_size[2],
                                       input_size[3],
                                       input_size[4]))

    conv1 = tf.nn.conv2d(data, gauss_op, [1, 1, 1, 1], padding='SAME')
    if len(input_size) == 5:
        out_size = conv1.shape.as_list()
        conv1 = tf.reshape(conv1, shape=(input_size[0],
                                         input_size[1],
                                         out_size[-3],
                                         out_size[-2],
                                         out_size[-1]))
    return conv1


def lap_loss(X,Y):
    return tf.reduce_mean(tf.abs(laplacian(X)-laplacian(Y)))

def sobel_loss(pred_data, gt):
    grad_pred_x=first_grad(pred_data,0)
    grad_pred_y=first_grad(pred_data,1)
    grad_gt_x=first_grad(gt,0)
    grad_gt_y=first_grad(gt,1)
    return tf.reduce_mean(tf.abs(grad_pred_x-grad_gt_x))+tf.reduce_mean(tf.abs(grad_pred_y-grad_gt_y))
def first_grad(data, axis=0):
    #在某个维度上算一阶梯度
    gauss_op = tf.constant(
        np.reshape((1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
                   [3, 3, 1, 1]), dtype=tf.float32)
    if axis == 0:
        sobel_op = tf.constant(np.reshape([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], [3, 3, 1, 1]), dtype=tf.float32)
    else:
        sobel_op = tf.constant(np.reshape([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], [3, 3, 1, 1]), dtype=tf.float32)
    input_size = data.shape.as_list()
    if len(input_size) == 5:
        data = tf.reshape(data, shape=(input_size[0] * input_size[1],
                                         input_size[2],
                                         input_size[3],
                                         input_size[4]))

    conv1 = tf.nn.conv2d(data, gauss_op, [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.conv2d(conv1, sobel_op, [1, 1, 1, 1], padding='SAME')
    if len(input_size) == 5:
        out_size = conv2.shape.as_list()
        conv2 = tf.reshape(conv2, shape=(input_size[0],
                                     input_size[1],
                                     out_size[-3],
                                     out_size[-2],
                                     out_size[-1]))
    return conv2



def ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value
def tf_multi_ssim(img1,img2,stage=2):
    h=img1.shape.as_list()[2]
    ssim=[]
    step=h//stage
    for k in range(stage):
        list_j = []
        for j in range(stage):
            start_x = k * step
            start_y = j * step
            ssim.append(
                tf_ssim(img1[:,:,start_x:start_x+step,start_y:start_y+step],
                        img2[:,:,start_x:start_x+step,start_y:start_y+step]
                        ))

    return tf.reduce_mean(tf.stack(ssim))+tf.reduce_mean(tf_ssim(img1,img2))


def tf_ssim(img1,img2):

    seq_length = img1.get_shape().as_list()[1]
    loss_ssim = []
    for t in range(seq_length):
        loss_ssim.append(ssim(img1[:,t],img2[:,t]))
    return 1-tf.reduce_mean(tf.stack(loss_ssim))
def config_log():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s \n %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(c.SAVE_PATH, "train.log"),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames = new_frames * 3 / 255
    # new_frames /= (255 / 2)
    # new_frames -= 1
    # if frames.min() == np.inf or frames.max() == np.inf or new_frames.min() == np.inf or new_frames.max() == np.inf:
    #     print(frames.min(), frames.max(), new_frames.min(), new_frames.max())
    return new_frames


def rainfall_to_pixel(rainfall_intensity, a=None, b=None):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    if a is None:
        a = c.ZR_a
    if b is None:
        b = c.ZR_b
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def get_loss_weight_symbol(data):
    if c.USE_BALANCED_LOSS:
        balancing_weights = c.BALANCING_WEIGHTS
        weights = tf.ones_like(data) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in c.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * tf.to_float(data >= threshold)
        weights = weights
    else:
        weights = tf.ones_like(data)

    return weights


def weighted_mse(pred, gt, weight):
    return weighted_l2(pred, gt, weight)


def weighted_l2(pred, gt, weight):
    """

    Parameters
    ----------
    pred : tensor
        Shape: (batch_size, seq_len, H, W, C)
    gt : tensor
        Shape: (batch_size, seq_len, H, W, C)
    weight : tensor
        Shape: (batch_size, seq_len, H, W, C)

    Returns
    -------
    l2 : Value
    """
    l2 = weight * tf.square(pred - gt)
    l2 = tf.reduce_sum(l2)
    return l2


def weighted_l1(pred, gt, weight):
    l1 = weight * tf.abs(pred - gt)
    l1 = tf.reduce_sum(l1)
    return l1


def weighted_mae(pred, gt, weight):
    return weighted_l1(pred, gt, weight)

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames / 3 * 255
    # new_frames = frames + 1
    # new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames = new_frames * 3 / 255
    # new_frames /= (255 / 2)
    # new_frames -= 1
    # if frames.min() == np.inf or frames.max() == np.inf or new_frames.min() == np.inf or new_frames.max() == np.inf:
    #     print(frames.min(), frames.max(), new_frames.min(), new_frames.max())
    return new_frames


def save_png(data, path):
    data[data < 0] = 0

    data = data.astype(np.uint8)

    if not os.path.exists(path):
        os.makedirs(path)
    shape = data.shape
    data = data.reshape(shape[0], shape[-3], shape[-2])
    i = 1
    for img in data[:]:
        # img = cv2.GaussianBlur(img, ksize=(3,3), dst=None, sigmaY=0,sigmaX=0)
        img = Image.fromarray(img)
        # img=cv2.GaussianBlur(img,ksize=[5,5],dst=None,sigmaY=0)
        img.save(os.path.join(path, str(i) + ".png"))
        i += 1


def one_step_diff(dat, axis):
    """

    Parameters
    ----------
    dat : tensor (b, length, h, w, c)
    axes : int 2, 3

    Returns
    -------

    """
    if axis == 2:
        return dat[:, :, :-1, :, :] - dat[:, :, 1:, :, :]
    elif axis == 3:
        return dat[:, :, :, :-1, :] - dat[:, :, :, 1:, :]
    else:
        raise NotImplementedError


def gdl_loss(pred, gt):
    """

    Parameters
    ----------
    pred : tensor
        Shape: (b, length, h, w, c)
    gt : tensor
        Shape: (b, length, h, w, c)
    Returns
    -------
    gdl : value
    """
    pred_diff_h = tf.abs(one_step_diff(pred, axis=2))
    pred_diff_w = tf.abs(one_step_diff(pred, axis=3))
    gt_diff_h = tf.abs(one_step_diff(gt, axis=2))
    gt_diff_w = tf.abs(one_step_diff(gt, axis=3))
    gd_h = tf.abs(pred_diff_h - gt_diff_h)
    gd_w = tf.abs(pred_diff_w - gt_diff_w)
    gdl = tf.reduce_sum(gd_h) + tf.reduce_sum(gd_w)
    return gdl


def read_image(index, file_dir):
    path = os.path.join(file_dir, str(index))
    # files=os.listdir(path)
    images = []
    for i in range(0, c.IN_SEQ + c.OUT_SEQ):
        f = os.path.join(path, str(i+1) + '.png')
        images.append(np.array(Img.open(f)))
    return np.array(images)


def save_pred(seq_data, pred, index, valid_dir, color=False):
    sava_dir = os.path.join(valid_dir, str(index))
    # if sava_dir.index('Valid') >=0:
    display_dir = sava_dir.replace('Valid', 'display')
# if sava_dir.index('Test') >=0:
    display_dir = display_dir.replace('Test', 'display')
    real_data = seq_data
    save_png(real_data[0:5], sava_dir + '/in/')
    save_png(real_data[5:], sava_dir + '/out/')
    save_png(pred, sava_dir + '/pred/')
    if color:
        multi_process_transfer(sava_dir + '/in/', display_dir + '/in/')
        multi_process_transfer(sava_dir + '/out/', display_dir + '/out/')
        multi_process_transfer(sava_dir + '/pred/', display_dir + '/pred/')


def multi_thread_transfer(dir_path, des_path):
    """
    Transfer all the gray level images into rgba color images from dir_path to des_path.
    Theoretically, this function can parallels transfer multiple images.
    :param dir_path: input image directory
    :param des_path: output directory
    :return:
    """
    imgs = os.listdir(dir_path)
    origin_paths = []
    des_paths = []
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    for img in imgs:
        origin_paths.append(os.path.join(dir_path, img))
        des_paths.append(os.path.join(des_path, img))

    future_objs = []
    for i in range(len(imgs)):
        obj = _imread_executor_pool.submit(transfer, origin_paths[i], des_paths[i])
        future_objs.append(obj)
    wait(future_objs)


def transfer(img_path, des_path):
    """Transform the input image to rgba mode and save to a specified destination
    Parameters
    ----------
    img_path : path to a image, this image must be a gray level image.
    des_path : the destination path.

    """
    img = scipy.misc.imread(img_path)
    if img.shape == (912, 912):
        img = mapping(img[106:-106, 6:-6])
    elif img.shape == (720, 900):
        img = mapping(img[10:-10, :])
    else:
        img = mapping(img)
    # img = mapping(img)

    img.save(des_path, "PNG")


def mapping(img):
    """Map each gray level pixel in origin image to RGBA space
    Parameter
    ---------
    img : ndarray (a gray level image)

    Returns
    ---------
    img : An Image object with RGBA mode

    """
    # color_map = form_color_map()
    h, w = img.shape
    new_img = np.zeros((h, w, 4), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = color_map[img[i, j]]
    # print("done")
    img = Image.fromarray(new_img, mode="RGBA")
    return img


def array2RGB(img, des_path):
    # 直接将img数组转成彩色图像
    # print(des_path)
    if img.shape == (900, 900):
        img = mapping(img[100:-100, :])
    else:
        img = mapping(img)
    # img = mapping(img)
    try:
        img.save(des_path, "PNG")
        return True
    except:
        return False


def multi_process_transfer(dir_path, des_path):
    """
    Transfer all the gray level images into rgba color images from dir_path to des_path
    using multiprocess which can highly speed up the transfer process.
    :param dir_path:
    :param des_path:
    :return:
    """
    print(dir_path + '------->' + des_path)
    imgs = os.listdir(dir_path)
    origin_paths = []
    des_paths = []
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    for img in imgs:
        origin_paths.append(os.path.join(dir_path, img))
        des_paths.append(os.path.join(des_path, img))

    p = Pool()
    for i in range(len(imgs)):
        p.apply_async(transfer, args=(origin_paths[i], des_paths[i],))
        print('transfer success from ' + origin_paths[i] + '--to---' + des_paths[i])
    p.close()
    p.join()


def array2GRAY(img, des_path):
    """Transform the input image array to gray mode and save to a specified destination
    Parameters
    ----------
    img_path : path to a image, this image must be a gray level image.
    des_path : the destination path.
    """
    if img.shape == (720, 900):
        new_img = np.zeros([900, 900], dtype=np.uint8)
        new_img[90:-90, :] = img
        img = new_img
    img = Image.fromarray(img, mode="L")
    try:
        img.save(des_path, "PNG")
        return True
    except:
        return False


def read_img(path, read_storage):
    img = np.asarray(Image.open(path))
    # img=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
    # img=cv2.GaussianBlur(a,(5,5),0)
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    if c.H==480:
        read_storage[:] = img[:]
    if c.H==912:
    # if img.shape[0] == 900&c.H!=900:
    #     # read_storage[2:-2,14:-14,:] = img[100:-100,:,:]
    #     index_h = int((c.H - 700) / 2)
    #     # index_w=int((c.W-900)/2)
    #     read_storage[index_h:-index_h, :] = img[100:-100, :, :]
    # else:
        read_storage[6:-6,6:-6] = img[:]
    if c.H_test==912:
    # if img.shape[0] == 900&c.H!=900:
    #     # read_storage[2:-2,14:-14,:] = img[100:-100,:,:]
    #     index_h = int((c.H - 700) / 2)
    #     # index_w=int((c.W-900)/2)
    #     read_storage[index_h:-index_h, :] = img[100:-100, :, :]
    # else:
        read_storage[6:-6,6:-6] = img[:]
    if c.H==240:
        read_storage[10:-10,10:-10] = img[:]


def quick_read_frames(path_list, im_h, im_w):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    im_h : height of image
    im_w : width of image

    Returns
    -------

    """
    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            print(path_list[i])
            raise IOError
    read_storage = np.zeros((img_num, im_h, im_w, 1), dtype=np.uint8)
    if img_num == 1:
        read_img(path_list[0], read_storage[0])
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(read_img, path_list[i], read_storage[i])
            future_objs.append(obj)
        wait(future_objs)
    return read_storage[...]
def hist_color_valid(name):
    # name=NAME
    from post_processing.histmatch import hist_match_reverse
    for i in range(0,1001):
        # if i not in [137,167,220,223,291,298,299,313]:
        #     continue
        index=str(i)
        print( os.path.join(c.BASE_PATH, 'Valid', name, str(index)))
        if not os.path.exists(os.path.join(c.BASE_PATH,'Valid',name,str(index))):
            print('不存在',os.path.join(c.BASE_PATH,'Valid',name,str(index)))
            continue
        Valid_in_path=os.path.join(c.BASE_PATH,'Valid',name,str(index),'in')
        Valid_hist_path=os.path.join(c.BASE_PATH,'Valid',name,str(index),'hist')
        if not os.path.exists(Valid_hist_path):
            os.makedirs(Valid_hist_path)
        display_hist_path=os.path.join(c.BASE_PATH,'display',name,str(index),'hist')
        display_in_path=os.path.join(c.BASE_PATH,'display',name,str(index),'in')
        Valid_pred_path=os.path.join(c.BASE_PATH,'Valid',name,str(index),'pred')
        display_pred_path=os.path.join(c.BASE_PATH,'display',name,str(index),'pred')
        Valid_out_path=os.path.join(c.BASE_PATH,'Valid',name,str(index),'out')
        display_out_path=os.path.join(c.BASE_PATH,'display',name,str(index),'out')
        target_img = Image.open(Valid_in_path+'/5.png')
        target_img_array=np.array(target_img)

        if len(target_img_array[target_img_array>45])<2000:
            print('continue',i)
            continue
        target_hist = np.array(target_img.histogram())
        target_hist[0:15] = 0
        for i in range(1, c.OUT_SEQ+1):
            img = np.array(Image.open(os.path.join(Valid_pred_path,'{}.png'.format(i))),dtype=np.uint8)
            # img[img <15] = 0
            # img[img >80] = 0

            hist_image = hist_match_reverse(img, target_hist)
            import cv2

            cv2.imwrite(Valid_hist_path+'/{}.png'.format(i), hist_image)

        multi_process_transfer(Valid_in_path,
                               display_in_path)
        multi_process_transfer(Valid_out_path,
                               display_out_path)
        multi_process_transfer(Valid_pred_path,
                               display_pred_path)
        multi_process_transfer(Valid_hist_path,
                               display_hist_path)
def hist_color_test(name):
    from post_processing.histmatch import hist_match
    for i in os.listdir(os.path.join(c.BASE_PATH,'Test',name)):
        index=str(i)

        Valid_in_path=os.path.join(c.BASE_PATH,'Test',name,str(index),'in')
        Valid_hist_path=os.path.join(c.BASE_PATH,'Test',name,str(index),'predict')

        if not os.path.exists(Valid_hist_path):
            os.makedirs(Valid_hist_path)
        display_hist_path=os.path.join(c.BASE_PATH,'display',name,str(index),'predict')

        display_in_path=os.path.join(c.BASE_PATH,'display',name,str(index),'in')
        # Valid_pred_path=os.path.join(c.BASE_PATH,'Test',name,str(index),'pred')
        # display_pred_path=os.path.join(c.BASE_PATH,'display',name,str(index),'pred')
        Valid_out_path=os.path.join(c.BASE_PATH,'Test',name,str(index),'out')
        display_out_path=os.path.join(c.BASE_PATH,'display',name,str(index),'out')
        # target_img = Image.open(Valid_in_path+'/5.png')
        # target_img_array=np.array(target_img)
        # if len(target_img_array[target_img_array>0])<60000 or len(target_img_array[target_img_array>45])<4000:
        #     print('continue',i)
        #     continue
        # target_hist = np.array(target_img.histogram())
        # target_hist[]
        # target_hist[0:15] = 0
        # for i in range(1, c.OUT_SEQ+1):
        #     # target_img_array = np.array(target_img)
        #     if i>10:
        #         target_hist = np.array(Image.fromarray(hist_image).histogram())
        #     target_hist[0:15] = 0
        #     img = np.array(Image.open(os.path.join(Valid_pred_path,'{}.png'.format(i))),dtype=np.uint8)
        #     # img[img < 15] = 0
        #     # img[img > 80] = 0
        #     hist_image = hist_match(img, target_hist)
        #     import cv2
        #
        #     cv2.imwrite(Valid_hist_path+'/{}.png'.format(i), hist_image)
        # multi_process_transfer(Valid_hist_path,
        #                        display_hist_path)
        # target_hist = np.array(target_img.histogram())
        # target_hist[]
        # target_hist[0:15] = 0
        # for i in range(1, c.OUT_SEQ + 1):
        #     # target_img_array = np.array(target_img)
        #     # if i > 10:
        #     #     target_hist = np.array(Image.fromarray(hist_image).histogram())
        #     # target_hist[0:15] = 0
        #     img = np.array(Image.open(os.path.join(Valid_pred_path, '{}.png'.format(i))), dtype=np.uint8)
        #     # img[img < 15] = 0
        #     # img[img > 80] = 0
        #     hist_image = hist_match(img, target_hist)
        #     import cv2
        #
        #     cv2.imwrite(Valid_oldhist_path + '/{}.png'.format(i), hist_image)
        #
        # multi_process_transfer(Valid_in_path,
        #                        display_in_path)
        # multi_process_transfer(Valid_out_path,
        #                        display_out_path)
        # multi_process_transfer(Valid_pred_path,
        #                        display_pred_path)
        multi_process_transfer(Valid_hist_path,
                               display_hist_path)
        # multi_process_transfer(Valid_oldhist_path,
        #                        display_oldhist_path)
def color_online():
    test_path='/extend/gru_tf_data/online_data/20200324_online'
    display_path='/extend/gru_tf_data/online_data/20200324color'
    for i in os.listdir(test_path):
        if i.startswith('2020032410') or i.startswith('2020032411') or i.startswith('2020032412')   :
            pred_gray_path=os.path.join(test_path,i)
            pred_display_path=os.path.join(display_path,i)
            multi_process_transfer(pred_gray_path,pred_display_path)
def getHist(data):
    # index=0
    hist=np.zeros(shape=[len(data),13],dtype=np.float)
    for i in range(len(data)):
        img=data[i]
        sum=len(img[img>0])
        for j in range(13):
            n1=len(img[img>(15+j*5)])
            n2=len(img[img>(20+j*5)])
            if sum==0:
                hist[i,j]=0
            else:
                if n1-n2==0:
                    hist[i, j] = sys.maxsize
                else:
                    hist[i,j]=(n1-n2)
    return np.sum(hist,axis=0)
if __name__ == '__main__':
    hist_color_valid('93999valid')
    # hist_color_test('404999')
    # pass
    # color_online()

