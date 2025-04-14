import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sn
import joblib

subj_path = 'D:\\Myo_Project\\MIO_trindets\\Fedor\\Raw\\preproc_angles\\1\\'

plt.close('all')

data = np.load(subj_path+'0000.npz')

plt.figure()
plt.plot(data)

print(data['myo_ts'].shape)
print(data['data_angles'].shape)
print(data['data_vr'].shape)
print(data['data_myo'].shape)
print(data['std_coef'])

fs = 500
f, pxx = sn.welch(x=data['data_myo'], axis=0, fs=fs, nperseg=int(fs*2))

plt.figure()
plt.plot(f, np.log10(pxx))

def train_test_split(data,N_parts,num_of_part):
    N_samples =len(data)
    
    l_idx = int((N_samples*num_of_part)/N_parts)
    h_idx = int((N_samples*(num_of_part+1))/N_parts)
    
    data_train = np.concatenate([data[:l_idx,:],data[h_idx:,]],axis = 0)
    data_test = data[l_idx:h_idx,:]
    
    return data_train, data_test

#%%
N_files = 4
data_list_train = list()
data_list_test = list()
label_list_train = list()
label_list_test = list()

N_parts = 10
num_of_part = 9

for i in range(N_files):
    
    arr = np.load(subj_path+'000'+str(i)+'.npz')
    
    std_coef = arr['std_coef']
    data = arr['data_myo']#[:,:6]
    label = arr['data_angles']
    
    #data -= np.mean(data,axis = 0)
    #label -= np.mean(label,axis = 0)
    
    
    data_train,data_test = train_test_split(data,N_parts,num_of_part)
    label_train,label_test = train_test_split(label,N_parts,num_of_part)
   
   
    data_list_train.append(data_train)
    data_list_test.append(data_test)
    
    label_list_train.append(label_train)
    label_list_test.append(label_test)
    
     
data_train = np.concatenate(data_list_train,axis = 0)
data_test = np.concatenate(data_list_test,axis = 0)

   
label_train = np.concatenate(label_list_train,axis = 0)
label_test = np.concatenate(label_list_test,axis = 0)

#%%
#  timestep in ms
#  window length in ms
# data inpute - time x channels 
# data output - epochs x channels x time
def slicer(data, label,fs, windowlen = 500, timestep = 100):
    data_len = len(data)
    timestep_samples = int((timestep*fs)/1000)
    windowlen_samples = int((windowlen*fs)/1000)
    start_idc = np.arange(0,data_len-windowlen_samples,timestep_samples)[:,None]
    window_idc = np.arange(0,windowlen_samples)[None,:]
    slice_idc = start_idc+window_idc
    slice_data = data[slice_idc].transpose(0,2,1)
    slice_label = label[start_idc[:,0]+windowlen_samples]
    return slice_data, slice_label
      
#%%

from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
    
X_train, y_train = slicer(data_train,label_train,fs)
X_test, y_test = slicer(data_test,label_test,fs)
 
pipeline1 = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LinearRegression()) #CatBoostRegressor())#logging_level='Silent'))
pipeline1.fit(X_train, y_train)

pipeline2 = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), MLPRegressor(hidden_layer_sizes=(200,), activation='relu', solver='adam', alpha=0.00001)) #CatBoostRegressor())#logging_level='Silent'))
pipeline2.fit(X_train, y_train)



pipeline3 = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), MLPRegressor(hidden_layer_sizes=(400,400), activation='relu', solver='adam', alpha=0.001)) #CatBoostRegressor())#logging_level='Silent'))
pipeline3.fit(X_train, y_train)

#params = {'learning_rate': 0.3, 
#          'depth': 6, 
#          'l2_leaf_reg': 3, 
#          'loss_function': 'MultiRMSE', 
#          'eval_metric': 'MultiRMSE'}

#pipeline3 = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), CatBoostRegressor(**params))
#pipeline3.fit(X_train, y_train)

#pipeline4 = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), MLPRegressor(hidden_layer_sizes=(400,), activation='relu', solver='adam', alpha=0.00001)) #CatBoostRegressor())#logging_level='Silent'))
#pipeline4.fit(X_train, y_train)


#%%
def corrcoef(x,y):
    x2 = x- np.mean(x)
    y2 = y-np.mean(y)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    return np.sum(x*y)/(normx*normy)

#%%

y_predict1 = pipeline1.predict(X_test)
y_predict2 = pipeline2.predict(X_test)
y_predict3 = pipeline3.predict(X_test)
#y_predict3 = pipeline3.predict(X_test)

for i in range(y_test.shape[1]):
    print(corrcoef(y_predict1[:,i],y_test[:,i]))
    print(corrcoef(y_predict2[:,i],y_test[:,i]))
    print(corrcoef(y_predict3[:,i],y_test[:,i]))
    #print(corrcoef(y_predict3[:,i],y_test[:,i]))
    print()

#%%



    




#%%


y_predict = pipeline2.predict(X_test)



plt.figure()
plt.plot(y_predict[int(100*fs/100):int(400*fs/100)])






#pipeline3.fit(X_train, y_train)


#%%


joblib.dump(pipeline3, 'real_time_regression/Fedor_TEST.pickle')
np.save('real_time_regression/preproc_params_Fedor_TEST.npy', std_coef)

#%%


for i in range(y_test.shape[1]):
    
    plt.figure()
    plt.plot(y_test[:,i])
    plt.plot(y_predict[:,i])
    

#%%
    

'''
ud,sd,vd = np.linalg.svd(data_train,full_matrices = False)
ul,sl,vl = np.linalg.svd(label_train,full_matrices = False)


plt.figure()
plt.plot(sd)
plt.plot(sl)


thr_sd = np.cumsum(sd)/np.sum(sd)
thr_sl = np.cumsum(sl)/np.sum(sl)


bool_thr_sd = thr_sd < 0.9
bool_thr_sl = thr_sl < 0.9



   

'''
    
    
    

    
    
    

    
    
    













































