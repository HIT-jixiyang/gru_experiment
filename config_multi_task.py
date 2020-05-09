import os


def config_gru_fms(height, strides):
    gru_fms = [height]
    for i, s in enumerate(strides):
        gru_fms.append(gru_fms[i] // s)
    return gru_fms[1:]


def config_deconv_infer_height(height, strides):
    infer_shape = [height]
    for i, s in enumerate(strides[:-1]):
        infer_shape.append(infer_shape[i] // s)
    return infer_shape


def config_deconv_infer_width(width, strides):
    infer_shape = [width]
    for i, s in enumerate(strides[:-1]):
        infer_shape.append(infer_shape[i] // s)
    return infer_shape


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


BASE_PATH = '/extend/gru_tf_data/gru_experiment/multi_task'
# BASE_PATH='/extend/gru_tf_data/gru_experiment/ln_multi_scale_221'
SAVE_PATH = os.path.join(BASE_PATH, 'Save')
SAVE_SUMMARY = os.path.join(BASE_PATH, 'Summary')
DISPLAY_PATH = os.path.join(BASE_PATH, 'display')
VALID_PATH = os.path.join(BASE_PATH, 'Valid')
TEST_PATH = os.path.join(BASE_PATH, 'Test')
SAVE_METRIC = os.path.join(BASE_PATH, 'Metrics')
check_dir(BASE_PATH)
check_dir(SAVE_PATH)
check_dir(SAVE_SUMMARY)
check_dir(DISPLAY_PATH)
check_dir(VALID_PATH)
check_dir(TEST_PATH)

TRAIN_DATA_PATH = '/extend/radar_crop_data_360/train'
VALID_DATA_PATH = '/extend/radar_crop_data_360/valid'
TEST_RADAR_PNG_PATH = '/extend/2019_png/'
# TEST_RADAR_PNG_PATH='/extend/2019_png'

TEST_STRIDE = 5
TEST_TIME = [

    # ['201903050000','201903060000'],
    # ['201903070000','201903080000'],
    ['201904110000', '201904120000'],
    ['201904190000', '201904191200'],
    ['201904200000', '201904201200'],
    ['201905270000', '201905271200'],
    ['201905070000', '201905080000'],
    ['201905210000', '201905220000'],
    ['201905230000', '201905240000'],
    ['201905270000', '201905300000'],
    ['201906010000', '201906020000'],
    ['201906110000', '201906130000'],
    ['201906240000', '201906260000'],
    ['201907030000', '201907040000'],
    ['201907100000', '201907110000'],
    ['201907310000', '201908010000'],
    ['201908010000', '201908020000'],
    ['201908110000', '201908120000'],
    ['201908180000', '201908190000'],
    ['201908250000', '201908270000'],
    ['201908310000', '201909010000'],
]
TRAIN_SEQ = 40000
VALID__SEQ = 1000
MIN_MAX_NORM = False
BATCH_SIZE = 2
W = 360
H = 360
OUT_SEQ = 20
IN_SEQ = 5
TEST_OUT_SEQ = 20
W_test = 912
H_test = 912
IN_CHANEL = 2
MAX_ITER = 250000
SAVE_ITER = 5000
VALID_ITER = 5000
SUMMARY_ITER = 5
CL = True
if CL:
    resize = 2
else:
    resize = 1
# generator--------------------------------
GEN_ENCODER_GRU_FILTER=(64,192,192)
GEN_ENCODER_GRU_INCHANEL=(8+resize,128+resize,192+resize)
GEN_CONV_KERNEL = ((7, 7, IN_CHANEL, 8),
               (5, 5, 64, 128),
               (3, 3, 192, 192))
GEN_CONV_STRIDE = (3, 2, 2)
GEN_DECONV_KERNEL = ((7, 7, 8, 64),
                 (5, 5, 128, 192),
                 (4, 4, 192, 192))
GEN_DECONV_STRIDE = (3, 2, 2)
GEN_DECODER_GRU_FILTER= (64, 192, 192)
GEN_DECODER_GRU_INCHANEL = (128, 192, 192)
GEN_I2H_KERNEL = [5, 3, 3]
GEN_H2H_KERNEL = [7, 5, 3]
GEN_LR = 0.0001
ZR_a = 58.53
ZR_b = 1.56
THRESHOLDS = [0.5, 2, 5, 10, 30]
EVALUATION_THRESHOLDS = (20, 30, 40, 50)
BALANCING_WEIGHTS = [1, 1, 2, 5, 10, 30]
# d_model--------------------------
D_ENCODER_GRU_FILTER = (64, 192, 192)
D_ENCODER_GRU_INCHANEL = (8, 192, 192)
D_CONV_KERNEL = ((7, 7, 2, 8),
                 (5, 5, 64, 192),
                 (3, 3, 192, 192))
D_CONV_STRIDE = (3, 2, 2)
D_DECONV_KERNEL = ((7, 7, 8, 64),
                   (5, 5, 64, 192),
                   (4, 4, 192, 192))
D_DECONV_STRIDE = (3, 2, 2)
D_DECODER_GRU_FILTER = (64, 192, 192)
D_DECODER_GRU_INCHANEL = (64, 192, 192)
D_I2H_KERNEL = [5, 3, 3]
D_H2H_KERNEL = [7, 5, 3]
D_LR = 0.0001
USE_BALANCED_LOSS = False
lam_l1 = 1
lam_l2 = 0
lam_grad1 = 0
lam_grad2 = 0
lam_max = 0
lam_gdl = 0
lam_ssim = 0
lam_cl = 1
LN = True
IN = False
lam_hinge = 0
multi_ln = False
LN_size = [
    [H // 12, H // 6],
    [H // 24, H // 12],
    [H // 24]

]
# [<tf.Variable 'generator/e_cgru_0_Wi:0' shape=(5, 5, 18, 288) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_0_bi:0' shape=(288,) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_0_Wh:0' shape=(7, 7, 96, 288) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_0_bh:0' shape=(288,) dtype=float32_ref>, <tf.Variable 'generator/Conv0_W:0' shape=(7, 7, 2, 16) dtype=float32_ref>, <tf.Variable 'generator/Conv0_b:0' shape=(16,) dtype=float32_ref>, <tf.Variable 'generator/EnState_conv0_W:0' shape=(1, 1, 96, 96) dtype=float32_ref>, <tf.Variable 'generator/EnState_conv0_b:0' shape=(96,) dtype=float32_ref>, <tf.Variable 'generator/DeState_conv0_W:0' shape=(1, 1, 96, 96) dtype=float32_ref>, <tf.Variable 'generator/DeState_conv0_b:0' shape=(96,) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_1_Wi:0' shape=(3, 3, 194, 576) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_1_bi:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_1_Wh:0' shape=(5, 5, 192, 576) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_1_bh:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/Conv1_W:0' shape=(5, 5, 96, 192) dtype=float32_ref>, <tf.Variable 'generator/Conv1_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/EnState_conv1_W:0' shape=(1, 1, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/EnState_conv1_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/DeState_conv1_W:0' shape=(1, 1, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/DeState_conv1_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_2_Wi:0' shape=(3, 3, 194, 576) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_2_bi:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_2_Wh:0' shape=(3, 3, 192, 576) dtype=float32_ref>, <tf.Variable 'generator/e_cgru_2_bh:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/Conv2_W:0' shape=(3, 3, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/Conv2_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/EnState_conv2_W:0' shape=(1, 1, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/EnState_conv2_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/DeState_conv2_W:0' shape=(1, 1, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/DeState_conv2_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/Final_conv1_W:0' shape=(7, 7, 16, 1) dtype=float32_ref>, <tf.Variable 'generator/Final_conv1_b:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'generator/CL_conv1_W:0' shape=(7, 7, 16, 17) dtype=float32_ref>, <tf.Variable 'generator/CL_conv1_b:0' shape=(17,) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_0_Wi:0' shape=(5, 5, 192, 288) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_0_bi:0' shape=(288,) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_0_Wh:0' shape=(7, 7, 96, 288) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_0_bh:0' shape=(288,) dtype=float32_ref>, <tf.Variable 'generator/Deconv0_W:0' shape=(7, 7, 16, 96) dtype=float32_ref>, <tf.Variable 'generator/Deconv0_b:0' shape=(16,) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_1_Wi:0' shape=(3, 3, 192, 576) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_1_bi:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_1_Wh:0' shape=(5, 5, 192, 576) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_1_bh:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/Deconv1_W:0' shape=(5, 5, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/Deconv1_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_2_Wi:0' shape=(3, 3, 192, 576) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_2_bi:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_2_Wh:0' shape=(3, 3, 192, 576) dtype=float32_ref>, <tf.Variable 'generator/f_cgru_2_bh:0' shape=(576,) dtype=float32_ref>, <tf.Variable 'generator/Deconv2_W:0' shape=(4, 4, 192, 192) dtype=float32_ref>, <tf.Variable 'generator/Deconv2_b:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Encoder/E_LN_0/ln_0/beta:0' shape=(96,) dtype=float32_ref>, <tf.Variable 'generator/AE/Encoder/E_LN_0/ln_0/gamma:0' shape=(96,) dtype=float32_ref>, <tf.Variable 'generator/AE/Encoder/E_LN_1/ln_1/beta:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Encoder/E_LN_1/ln_1/gamma:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Encoder/E_LN_2/ln_2/beta:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Encoder/E_LN_2/ln_2/gamma:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Forecaster/D_LN_2/ln_2/beta:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Forecaster/D_LN_2/ln_2/gamma:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Forecaster/D_LN_1/ln_1/beta:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Forecaster/D_LN_1/ln_1/gamma:0' shape=(192,) dtype=float32_ref>, <tf.Variable 'generator/AE/Forecaster/D_LN_0/ln_0/beta:0' shape=(96,) dtype=float32_ref>, <tf.Variable 'generator/AE/Forecaster/D_LN_0/ln_0/gamma:0' shape=(96,) dtype=float32_ref>]
