import numpy as np
from PIL import Image


# 直方图匹配函数，接受原始图像和目标灰度直方图 orig_img[700,900],tgt_hist[256]
def hist_match(orig_img, tgt_hist):
    orig_hist = np.array(Image.fromarray(orig_img).histogram())
    orig_hist[0] = 0
    # 分别对orig_acc和tgt_acc归一化
    orig_sum = 0.0
    tgt_sum = 0.0
    for i in range(1, 80):
        orig_sum += orig_hist[i]
        tgt_sum += tgt_hist[i]
    # for i in range(1, 80):
    orig_hist= orig_hist/(orig_sum+1)
    tgt_hist=tgt_hist/ tgt_sum
    # 计算累计直方图
    tmp = 0.0
    tgt_acc = tgt_hist.copy()
    for i in range(1, 256):
        tmp += tgt_hist[i]
        tgt_acc[i] = tmp
    tmp = 0.0
    orig_acc = orig_hist.copy()
    for i in range(1, 256):
        tmp += orig_hist[i]
        orig_acc[i] = tmp

    # 计算映射
    M = np.zeros(256,dtype=np.uint8)
    for i in range(1, 256):
        idx = 16
        minv = 1
        for j in range(1, 256):
            if np.fabs(tgt_acc[j] - orig_acc[i]) < minv:
                # update the value of minv
                minv = np.fabs(tgt_acc[j] - orig_acc[i])
                idx = int(j)
        # if idx-M[i-1]>2:
        #     M[i]=M[i-1]+1
        # else:
        M[i] = idx
        # M stores the index of closest tgt_hist gray value
    # print(M)
    orig_img=orig_img.astype(np.int)
    des = M[orig_img]
    return des


# 直方图匹配函数，接受原始图像和目标灰度直方图 orig_img[700,900],tgt_hist[256]
def hist_match_reverse(orig_img, tgt_hist):
    input_max=80
    for i in range(len(tgt_hist) - 1, -1, -1):
        if tgt_hist[i] > 5:
            input_max = i
            break
    orig_hist = np.array(Image.fromarray(orig_img).histogram())
    orig_hist[0] = 0
    # 分别对orig_acc和tgt_acc归一化
    orig_sum = 0.0
    tgt_sum = 0.0
    for i in range(1, input_max):
        orig_sum += orig_hist[i]
        tgt_sum += tgt_hist[i]
    # for i in range(1, 80):
    orig_hist= orig_hist/(orig_sum+1)
    tgt_hist=tgt_hist/ tgt_sum
    # 计算累计直方图
    tmp = 0.0
    tgt_acc = tgt_hist.copy()
    for i in range(input_max, 0,-1):
        tmp += tgt_hist[i]
        tgt_acc[i] = tmp
    tmp = 0.0
    orig_acc = orig_hist.copy()
    for i in range(input_max,0, -1):
        tmp += orig_hist[i]
        orig_acc[i] = tmp

    # 计算映射
    M = np.zeros(256,dtype=np.uint8)
    for i in range(input_max, 0,-1):
        idx = input_max
        minv = 1
        for j in range(input_max, 0,-1):
            if np.fabs(tgt_acc[j] - orig_acc[i]) < minv:
                # update the value of minv
                minv = np.fabs(tgt_acc[j] - orig_acc[i])
                idx = int(j)
        # if idx-M[i-1]>2:
        #     M[i]=M[i-1]+1
        # else:
        M[i] = idx
        # M stores the index of closest tgt_hist gray value
    # print(M)
    orig_img=orig_img.astype(np.int)
    des = M[orig_img]
    return des

def array2dict(array):
    hist_dict = {}
    for i in range(array.shape[0]):
        hist_dict[i] = array[i]
    return hist_dict


def hist_pred(pred_result, pred_hist):
    pred_result = np.reshape(pred_result, [pred_result.shape[0], pred_result.shape[1], pred_result.shape[2]])
    # print(pred_result.shape)
    # print(pred_hist.shape)
    for i in range(pred_result.shape[0]):
        hist_dict = array2dict(pred_hist[i])
        des=hist_match(pred_result[i], hist_dict)
        pred_result[i] = des
    return np.reshape(pred_result, [pred_result.shape[0], pred_result.shape[1], pred_result.shape[2],1])


if __name__ == '__main__':
    target_img=Image.open('/extend/gru_tf_data/began/Test/49999test/0/in/5.png')
    target_hist=np.array(target_img.histogram())
    target_hist[0]=0
    for i in range(1,11):

        img=np.array(Image.open('/extend/gru_tf_data/began/Test/49999test/0/pred/{}.png'.format(i)),dtype=np.uint8)
        img[img<15]=0
        img[img>80]=0

        hist_image=hist_match(img,target_hist)
        import cv2
        cv2.imwrite('/extend/gru_tf_data/began/Test/49999test/0/hist/{}.png'.format(i),hist_image)



