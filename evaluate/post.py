import os
from color_map import multi_process_transfer
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
# from evaluation import Evaluator
# evaluator = Evaluator(os.path.join('/extend/output/evaluation-2019-09/iter300000_test_finetune1/metrics'))
path = '/extend/gru_tf_data/gru_norelu/Test/49999'
time = ['201803010600', '201810181200']
time_range = pd.date_range(start=time[0], end=time[1], freq='6min')
import shutil as sh
length=10

from histmatch import hist_match_reverse as hist_match
def hist_preprocessing(target_hist):
    input_max = 0
    for i in range(len(target_hist) - 1, -1, -1):
        if target_hist[i] > 5:
            input_max = i
            break

    target_hist[input_max+1:] = 0
    return target_hist

def hist_by_in_5(Valid_pred_path,Valid_hist_path,target_hist_5):
    print('使用统一规定化方案')
    target_hist_5[0:14] = 0
    target_hist_5=hist_preprocessing(target_hist_5)
    for i in range(1, length+1):
        # target_img_array = np.array(target_img)
        # if i>10:
        #     target_hist = np.array(Image.fromarray(hist_image).histogram())

        img = np.array(Image.open(os.path.join(Valid_pred_path, '{}.png'.format(i))), dtype=np.uint8)
        # img[img < 15] = 0
        # img[img > 80] = 0
        hist_image = hist_match(img, target_hist_5)


        cv2.imwrite(Valid_hist_path + '/{}.png'.format(i), hist_image)
        # print(i)
def hist_cascade(Valid_pred_path,Valid_hist_path,target_hist_5):
    print('使用级联规定化方案')
    target_hist_5[0:15] = 0


    for i in range(1, length+1):

        if i>1:
            target_hist_5 = (1/3*np.array(Image.fromarray(hist_image).histogram())+2/3*target_hist_5)//2
        target_hist_5=hist_preprocessing(target_hist_5)
        img = np.array(Image.open(os.path.join(Valid_pred_path, '{}.png'.format(i))), dtype=np.uint8)
        # img[img < 15] = 0
        # img[img > 80] = 0
        hist_image = hist_match(img, target_hist_5)

        cv2.imwrite(Valid_hist_path + '/{}.png'.format(i), hist_image)
        # print(i)
def hist_cascade_two_stage(Valid_pred_path,Valid_hist_path,target_hist_5):
    print('使用两段规定化方案')
    target_hist_5[0:15] = 0

    for i in range(1, length+1):

        if i>10:
            target_hist_5 = (1/3*np.array(Image.fromarray(hist_image_10).histogram())+2/3*target_hist_5)//2

            target_hist_5=hist_preprocessing(target_hist_5)
            img = np.array(Image.open(os.path.join(Valid_pred_path, '{}.png'.format(i))), dtype=np.uint8)
            # img[img < 15] = 0
            # img[img > 80] = 0
            hist_image = hist_match(img, target_hist_5)

        if i<=10:
            target_hist_5 = hist_preprocessing(target_hist_5)

            img = np.array(Image.open(os.path.join(Valid_pred_path, '{}.png'.format(i))), dtype=np.uint8)
            # img[img < 15] = 0
            # img[img > 80] = 0
            hist_image = hist_match(img, target_hist_5)
            if i==10:
                hist_image_10=hist_image

        cv2.imwrite(Valid_hist_path + '/{}.png'.format(i), hist_image)
        # print(i)


for i in range(len(time_range)):
    index = time_range[i].strftime("%Y%m%d%H%M")
    print(index)
    # index='201904201000'
    if not os.path.exists(os.path.join(path, str(index))):
        continue
    Valid_out_path = os.path.join(path, str(index), 'out')
    if not os.path.exists(Valid_out_path):
        continue
    Valid_hist_path = os.path.join(path, str(index), 'predict')
    # Valid_oldhist_path=os.path.join(c.BASE_PATH,'Test',name,str(index),'old_hist')
    if not os.path.exists(Valid_hist_path):
        os.makedirs(Valid_hist_path)

    Valid_pred_path = os.path.join(path, str(index), 'pred')

    target_img_5 = Image.open(Valid_out_path + '/1.png')
    # target_img_4 = Image.open(Valid_in_path + '/4.png')
    # target_img_3 = Image.open(Valid_in_path + '/3.png')

    target_img_array = np.array(target_img_5)
    # if len(target_img_array[target_img_array>0])<60000 or len(target_img_array[target_img_array>45])<4000:
    #     print('continue',i)
    #     continue
    target_hist_5 = np.array(target_img_5.histogram())
    # target_hist_4 = np.array(target_img_4.histogram())
    # target_hist_3 = np.array(target_img_3.histogram())
    try:
        hist_by_in_5(Valid_pred_path, Valid_hist_path, target_hist_5)
        # if sum(target_hist_5[50:])>1000 and sum(target_hist_4[50:])>1000 and sum(target_hist_3[50:])>1000:
        #     hist_by_in_5(Valid_pred_path,Valid_hist_path,target_hist_5)
        # elif sum(target_hist_5[50:])>sum(target_hist_4[50:]) and sum(target_hist_4[50:])>sum(target_hist_3[50:]) and sum(target_hist_5[50:])>800:
        #     hist_by_in_5(Valid_pred_path,Valid_hist_path,target_hist_5)
        #
        # # elif sum(target_hist_5[50:])<sum(target_hist_4[50:]) and sum(target_hist_4[50:])<sum(target_hist_3[50:]):
        # #     hist_cascade(Valid_pred_path,Valid_hist_path,target_hist_5)
        # else:
        #     hist_cascade(Valid_pred_path,Valid_hist_path,target_hist_5)
        # hist_by_in_5(Valid_pred_path, Valid_hist_path, target_hist_5)
    except Exception as e:
        print(e)
        continue
    # target_hist[]


#
# for i in range(len(time_range)):
#     date = time_range[i].strftime("%Y%m%d%H%M")
#     save_path = os.path.join(path, date)
#     if os.path.exists(save_path):
#         # sh.copytree(save_path+'/pred',save_path+'/origin')
#         target_png=os.path.join(save_path,'in','5.png')
#         des_png=os.path.join(save_path,'in','10.png')
#         if not os.path.exists(des_png):
#             os.rename(target_png,des_png)
#         os.system(r'./postprocessing' + ' ' + os.path.join(save_path))
#         # display_path=os.path.join('/extend/gru_tf_data/began_norm1104/display/109999test',date)
#         # save_path=os.path.join(path,date)
#         # multi_process_transfer(save_path+'/in',display_path+'/in')
#         # multi_process_transfer(save_path+'/out',display_path+'/out')
#         # multi_process_transfer(save_path+'/pred',display_path+'/pred')
#         print('done',save_path)
