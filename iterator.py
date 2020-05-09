import os
from utils import *
import pandas as pd
class Iterator():

    def __init__(self,batch_size,test_time=c.TEST_TIME[0],test_stride=c.TEST_STRIDE):
        self.batch_size=batch_size
        self.test_time=test_time
        self.test_stride=test_stride
        start = self.test_time[0]
        end = self.test_time[1]
        freq = 6 * self.test_stride
        self.index = 0
        self.count1 = 0
        self.count2 = 0
        self.count_valid = 0
        self.use_up = False
        self.test_count = 0
        self.date_times = pd.date_range(start=start, end=end, freq=str(freq) + 'T')
        self.eval_dates=np.load('./eval_dates.npy')
        self.eval_counts=0
    def get_batch(self):
        data=[]
        try:
            for i in range(self.batch_size):
                # index=np.random.randint(0, 40000,1)[0]
                self.index=self.index+1
                if self.index>40000:
                    self.index=0
                    return None

                data.append(read_image(self.index, c.TRAIN_DATA_PATH))
            if len(data)==self.batch_size:
                return np.array(data)
        except Exception as e:
            # print(e.with_traceback())
            return None
    def get_valid_batch(self):
        data=[]
        indexes=[]
        if self.count_valid+self.batch_size>1000:
            self.use_up=True
            self.count_valid=0
            return None,None
        try:
            for i in range(self.batch_size):
                self.count_valid = self.count_valid + 1
                # index = np.random.randint(0, 2000, 1)[0]
                data.append(read_image(self.count_valid, c.VALID_DATA_PATH))
                indexes.append(self.count_valid)

            return np.array(data),indexes
        except Exception as e:
            print(e.with_traceback())
            return None
    def get_test_batch(self):
        print('total ',self.test_count,len(self.date_times))
        if len(self.date_times)-self.test_count<=1:
            return None,None
        test_data=np.zeros([self.batch_size,c.IN_SEQ+c.OUT_SEQ,c.H_test,c.W_test,1])
        date_clips=[]
        for i in range(self.test_count,self.test_count+self.batch_size):
            date_paths=[]
            date_clips1=pd.date_range(end=self.date_times[i],periods=c.IN_SEQ,freq='6T')
            date_clips2=pd.date_range(start=self.date_times[i],periods=c.OUT_SEQ+1,freq='6T')
            date_str1=date_clips1.strftime("%Y%m%d%H%M")
            date_str2=date_clips2.strftime("%Y%m%d%H%M")
            date_str=np.concatenate((date_str1,date_str2[1:]))
            date_clips.append(date_str)
            for j in range(0,c.IN_SEQ):
                date_paths.append(self.convert_datetime_to_filepath(date_clips1[j]))
            for j in range(1,c.OUT_SEQ+1):
                date_paths.append(self.convert_datetime_to_filepath(date_clips2[j]))
            try:
                all_frame_dat = quick_read_frames(date_paths,c.H_test,c.W_test)
            except OSError:
                return None,None
            test_data[i-self.test_count]=all_frame_dat
        self.test_count+=self.batch_size
        return test_data,date_clips
    def get_eval_batch(self):
        date_clips = []
        date_paths = []
        eval_data = np.zeros([1, c.IN_SEQ + c.OUT_SEQ, c.H_test, c.W_test, 1])
        if self.eval_counts>len(self.eval_dates)-1:
            return None, None

        date_clips1 = pd.date_range(end=self.eval_dates[self.eval_counts], periods=c.IN_SEQ, freq='6T')
        date_clips2 = pd.date_range(start=self.eval_dates[self.eval_counts], periods=c.OUT_SEQ + 1, freq='6T')
        date_str1 = date_clips1.strftime("%Y%m%d%H%M")
        date_str2 = date_clips2.strftime("%Y%m%d%H%M")
        date_str = np.concatenate((date_str1, date_str2[1:]))
        date_clips.append(date_str)
        for j in range(0, c.IN_SEQ):
            date_paths.append(self.convert_datetime_to_filepath(date_clips1[j]))
        for j in range(1, c.OUT_SEQ + 1):
            date_paths.append(self.convert_datetime_to_filepath(date_clips2[j]))
        self.eval_counts += 1
        try:
            all_frame_dat = quick_read_frames(date_paths, c.H_test, c.W_test)

        except OSError:
            return None, self.eval_dates[self.eval_counts-1]
        eval_data[0]=all_frame_dat

        return eval_data, date_clips
    def convert_datetime_to_filepath(self,date_time):
        """Convert datetime to the filepath

        Parameters
        ----------
        date_time : datetime.datetime

        Returns
        -------
        ret : str
        """
        # ret = os.path.join(cfg.REF_PATH, "cappi_ref_"+"%04d%02d%02d%02d%02d" %(
        #     date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute)
        #     +"_2500_0.ref")
        date_str = date_time.strftime("%Y%m%d%H%M")
        m = str(int(date_str[4:6]))
        ret = os.path.join(c.TEST_RADAR_PNG_PATH, "cappi_ref_" + date_time.strftime("%Y%m%d%H%M")
                           + "_2500_0.png")
        return ret
    def reset_count(self):
        self.count1=0
        self.count2=0

if __name__ == '__main__':
    iterator=Iterator(1)

    data= iterator.get_batch()
    print(data.shape)
    data = np.reshape(data, [1*25*360*360])
    data=(data+2)//5
    logit=np.zeros([1*25*360*360,80//5+1])
    for i in range(len(data)):
        logit[i,data[i]]=1
    logit=np.reshape(logit,[1,25,360,360,17])

    print(np.max(data))
