import os
import  numpy as np
import cv2
# month=range(10,11)
for m in range(3,4):
    ref_path='/extend/cappi/2020/'+str(m)+'/'
    list=os.listdir(ref_path)
    list=sorted(list)
    print(list[0])
    # try:
    #     temp_date=list[0].split('_')[2]
    # except:
    #     continue
    i=0
    date_radar_dict = {}
    while i<len(list)-1:
        try:
            date_current=list[i].split('_')[2]

        except:
            i = i + 1
            continue
        if not list[i].startswith('cappi_ref_20200318'):
            i = i + 1
            continue
        j = i + 1
        date_next=list[j].split('_')[2]
        radar_num_current=int(list[i].split('_')[4].split('.')[0])
        radar_num_next=int(list[j].split('_')[4].split('.')[0])
        temp_list=[]
        date_radar_dict[date_current] = radar_num_current
        if date_current==date_next:
            while date_current==date_next:
                if radar_num_next>radar_num_current:
                    date_radar_dict[date_current]=radar_num_next
                else:
                    date_radar_dict[date_current]=radar_num_current
                j= j+ 1
                i=j
                if i>len(list)-1:
                    break
                # date_current = list[i].split('_')[2]
                date_next = list[j].split('_')[2]
                # radar_num_current = list[i].split('_')[4].split('.')[0]
                radar_num_next = int(list[j].split('_')[4].split('.')[0])

        else:
            i=j
    for key in date_radar_dict.keys():
        path=ref_path+'/cappi_ref_'+key+'_2500_'+str(date_radar_dict[key])+'.ref'
        pad_ref = np.zeros([900, 900], dtype=np.uint8)
        try:
            ref = np.fromfile(path, dtype=np.uint8).reshape(700, 900)

        except:
            continue
        ref[ref <= 15] = 0
        ref[ref >= 80] = 0
        # if len(ref[ref >0])<3000:
        #     continue
        pad_ref[100:-100, :] = ref
        folder='/extend/radarPNG20200318/2020/'+str(m)+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(folder+'/cappi_ref_'+key+'_2500_'+str(0)+'.png',pad_ref)
        print(folder+'/cappi_ref_'+key+'_2500_'+str(0)+'.png')


# print()
