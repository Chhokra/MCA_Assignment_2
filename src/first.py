from scipy.io import wavfile
from math import pi
import numpy as np
sample_rate,signal = wavfile.read('training/eight/004ae714_nohash_0.wav')
from math import log,pow
import matplotlib.pyplot as plt
from librosa import display
import pickle
import os 
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def DFT(signal):
	result = np.zeros(len(signal),dtype=complex)
	for i in range(len(signal)):
		for j in range(len(signal)):
			result[i] = result[i]+signal[j]*np.exp(-1j*2*pi*i*j/len(signal))
	return result


def FFT(signal):
	if(len(signal)==1):
		return signal
	else:
		Even_Signal = np.zeros(len(signal)//2,dtype=complex)
		Odd_Signal = np.zeros(len(signal)//2,dtype=complex)
		for i in range(len(signal)):
			if(i%2==0):
				Even_Signal[i//2] = signal[i]
			else:
				Odd_Signal[(i-1)//2] = signal[i]
		Left_FFT = FFT(Even_Signal)
		Right_FFT = FFT(Odd_Signal)
		Final_FFT = np.zeros(len(signal),dtype=complex)
		for i in range(0,len(signal)//2):
			Final_FFT[i] = Left_FFT[i] + Right_FFT[i]*np.exp(-2j*pi*i/len(signal))
			Final_FFT[i+(len(signal)//2)] = Left_FFT[i] - Right_FFT[i]*np.exp(-2j*pi*i/len(signal))
		return Final_FFT

def padSignal(signal):
	if(log(len(signal),2)-int(log(len(signal),2))==0):
		return signal 
	else:
		new_length = pow(2,int(log(len(signal),2))+1)
		new_signal = np.zeros(int(new_length),dtype=complex)
		for i in range(len(signal)):
			new_signal[i] = signal[i]
		for i  in range(len(signal),len(new_signal)):
			new_signal[i] = 0.0000001
		return new_signal




def MakeSpectrogram(file_path,window_size,flag):
	Spectrogram = []
	sample_rate,signal = wavfile.read(file_path)
	signal = padSignal(signal)
	if(flag==1):
		random_number = random.randint(0,5)
		lis = []
		for (root,dirs,files) in os.walk('_background_noise_'):
			for file in files:
				lis.append(root+"/"+file)
		noise_file = lis[random_number]
		inner_sample_rate,inner_signal = wavfile.read(noise_file)
		random_number = random.randint(0,len(inner_signal)-len(signal)+1)
		noise_slice = inner_signal[random_number:random_number+len(signal)]
		signal = signal + 0.001*noise_slice



	if(log(window_size,2)-int(log(window_size,2))!=0):
		window_size = int(pow(2,int(log(window_size,2))+1))

	absolute_overlap = int(window_size*0.5)
	window_boundaries = []
	time = []
	for i in range(0,len(signal)+1,absolute_overlap):
		window_boundaries.append(i)
	for i in range(0,len(window_boundaries)-2):
		start = window_boundaries[i]
		end = window_boundaries[i+2]
		time.append(start)
		window_signal = signal[start:end]
		inner_FFT = FFT(window_signal)
		Spectrogram.append(np.abs(inner_FFT))
	Spectrogram = np.array(Spectrogram)
	return Spectrogram
	time = np.array(time)
	display.specshow(Spectrogram,x_coords=time,sr=sample_rate)
	plt.colorbar(format='%+2.0f dB')
	plt.title('Spectrogram')
	plt.show()



def MakePickles():
	for (root,dirs,files) in os.walk('training'):
		for file in files:
			string = root+"/"+file
			lis = string.split("/")
			file_handle = open('pickles/training/'+lis[1]+'/'+file.rstrip('.wav'),'wb')
			pickle.dump(MakeSpectrogram(root+'/'+file,256,1),file_handle)
	print("done")


def TrainSVM():
	svc = SVC()
	X_train = []
	Y_train = []
	for (root,dirs,files) in os.walk('pickles/training'):
		for file in files:
			file_path = root+'/'+file 
			element = pickle.load(open(file_path,'rb'))
			if(element.shape[0]==127):
				element = element.ravel()
				X_train.append(element)
				num_string = file_path.split('/')[2]
				print(num_string)
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
	file_handle = open('model.pkl','wb')
	pickle.dump(svc,file_handle)
	print("done")

def TestSVM(model_path):
	svm = pickle.load(open(model_path,'rb'))
	Y_Test = []
	X_Test = []
	for (root,dirs,files) in os.walk('pickles/validation'):
		for file in files:
			file_path = root+"/"+file 
			element = pickle.load(open(file_path,'rb'))
			if(element.shape[0]==127):
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

	file_handle1 = open('Spectrogram_Prediction','wb')
	file_handle2 = open('Spectrogram_GroundTruth','wb')
	pickle.dump(Y_Result,file_handle1)
	pickle.dump(Y_Test_For_Real,file_handle2)
	print("Done")


			

TestSVM('model.pkl')