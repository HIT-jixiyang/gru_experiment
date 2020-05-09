import numpy as np
var=[]
mean=[]
max=[]
min=[]
d_f=np.load('/home/ices/work/jxy/gru_experiment/中间状态/multi_ln/en_f_201904112100.npy')
for i in range(len(d_f)):
    # print('----------------------{}--------------------------'.format(i))

    var.append(str(np.var(d_f[i])))
    mean.append( str(np.mean(d_f[i])))
    max.append(str(np.max(d_f[i])))
    min.append(str(np.min(d_f[i])))


d_f=np.load('/home/ices/work/jxy/gru_experiment/中间状态/multi_ln/de_f_201904112100.npy')

for i in range(len(d_f)):
    # print('----------------------{}--------------------------'.format(i))

    var.append(str(np.var(d_f[i])))
    mean.append( str(np.mean(d_f[i])))
    max.append(str(np.max(d_f[i])))
    min.append(str(np.min(d_f[i])))
print(','.join((var)))
print(','.join(mean))
print(','.join(max))
print(','.join(min))
var=[]
mean=[]
max=[]
min=[]
d_f=np.load('/home/ices/work/jxy/gru_experiment/中间状态/multi_ln/en_f_201904112200.npy')
for i in range(len(d_f)):
    # print('----------------------{}--------------------------'.format(i))

    var.append(str(np.var(d_f[i])))
    mean.append( str(np.mean(d_f[i])))
    max.append(str(np.max(d_f[i])))
    min.append(str(np.min(d_f[i])))


d_f=np.load('/home/ices/work/jxy/gru_experiment/中间状态/multi_ln/de_f_201904112200.npy')

for i in range(len(d_f)):
    # print('----------------------{}--------------------------'.format(i))

    var.append(str(np.var(d_f[i])))
    mean.append( str(np.mean(d_f[i])))
    max.append(str(np.max(d_f[i])))
    min.append(str(np.min(d_f[i])))
print(','.join((var)))
print(','.join(mean))
print(','.join(max))
print(','.join(min))



