import numpy as np 
import os
import pickle 
import math
from scipy import fftpack
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def GetFilterBank(min_frequency,max_frequency,num_filter,sample_rate,window_size):

	FilterBank = [[0 for i in range((window_size//2)-1)] for j in range(num_filter)]
	max_in_mel = 1127*math.log((max_frequency/700)+1)
	min_in_mel = 1127*math.log((min_frequency/700)+1)

	get_uniform_mel = np.linspace(min_in_mel,max_in_mel,num=num_filter+2)
	get_uniform_frequency = 700*(2**(get_uniform_mel/1125))-1
	frequency_to_bin = []


	for i in range(len(get_uniform_frequency)):
		frequency_to_bin.append(math.floor(((window_size-1)/sample_rate)*get_uniform_frequency[i]))
	
	for i in range(len(frequency_to_bin)-2):
		for j in range(frequency_to_bin[i],frequency_to_bin[i+1]):
		
			FilterBank[i][j] = (j-frequency_to_bin[i])/(frequency_to_bin[i+1]-frequency_to_bin[i])
		for j in range(frequency_to_bin[i+1],frequency_to_bin[i+2]):
		
			FilterBank[i][j] = frequency_to_bin[i+2]-j/(frequency_to_bin[i+2]-frequency_to_bin[i+1])

	FilterBank = np.array(FilterBank)

	return FilterBank




def MakeMFCC(Spectrogram):
	
	power_spectrum = np.square(np.abs(Spectrogram))/256
	FilterBank = GetFilterBank(0,8000,20,16000,256)
	result = fftpack.dct(np.dot(FilterBank,Spectrogram))
	return result
	

def TrainSVM():
	svc = SVC()
	X_train = []
	Y_train = []
	for (root,dirs,files) in os.walk('pickles2/training'):
		for file in files:
			file_path = root+'/'+file 
			element = pickle.load(open(file_path,'rb'))
			element = element.ravel()
			X_train.append(element)
			num_string = file_path.split('/')[2]
			if(num_string=='zero'):
				Y_train.append(0)
			if(num_string=='one'):
				Y_train.append(1)
			if(num_string=='two'):
				Y_train.append(2)
			if(num_string=='three'):
				Y_train.append(3)
			if(num_string=='four'):
				Y_train.append(4)
			if(num_string=='five'):
				Y_train.append(5)
			if(num_string=='six'):
				Y_train.append(6)
			if(num_string=='seven'):
				Y_train.append(7)
			if(num_string=='eight'):
				Y_train.append(8)
			if(num_string=='nine'):
				Y_train.append(9)
			

	print("Training starts")
	svc.fit(X_train,Y_train)
	file_handle = open('model2.pkl','wb')
	pickle.dump(svc,file_handle)
	print("done")

def TestSVM(model_path):
	svm = pickle.load(open(model_path,'rb'))
	Y_Test = []
	X_Test = []
	for (root,dirs,files) in os.walk('pickles2/validation'):
		for file in files:
			file_path = root+"/"+file 
			element = pickle.load(open(file_path,'rb'))
			element = element.ravel()
			X_Test.append(element)
			Y_Test.append(file_path.split('/')[2])
	Y_Result = svm.predict(X_Test)
	Y_Test_For_Real = []
	for num_string in Y_Test:
		if(num_string=='zero'):
			Y_Test_For_Real.append(0)
		if(num_string=='one'):
			Y_Test_For_Real.append(1)
		if(num_string=='two'):
			Y_Test_For_Real.append(2)
		if(num_string=='three'):
			Y_Test_For_Real.append(3)
		if(num_string=='four'):
			Y_Test_For_Real.append(4)
		if(num_string=='five'):
			Y_Test_For_Real.append(5)
		if(num_string=='six'):
			Y_Test_For_Real.append(6)
		if(num_string=='seven'):
			Y_Test_For_Real.append(7)
		if(num_string=='eight'):
			Y_Test_For_Real.append(8)
		if(num_string=='nine'):
			Y_Test_For_Real.append(9)

	file_handle1 = open('MFCC_Prediction','wb')
	file_handle2 = open('MFCC_GroundTruth','wb')
	print(accuracy_score(Y_Result,Y_Test_For_Real))
	pickle.dump(Y_Result,file_handle1)
	pickle.dump(Y_Test_For_Real,file_handle2)
	print("Done")


	


# for (root,dirs,files) in os.walk('pickles/validation'):
# 	for file in files:
# 		file_name = root+"/"+file
# 		Spectrogram = pickle.load(open(root+"/"+file,'rb'))
# 		if(Spectrogram.shape[0]==127):	
# 			element = MakeMFCC(Spectrogram)
# 			file_handle = open('pickles2/validation/'+file_name.split("/")[2]+"/"+file,"wb")
# 			pickle.dump(element,file_handle)
# 			print("hi")

TestSVM('model2.pkl')
