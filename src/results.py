from sklearn.metrics import accuracy_score,classification_report 
import pickle

Spec_GroundTruth = pickle.load(open('MFCC_GroundTruth','rb'))
Spec_Prediction = pickle.load(open('MFCC_Prediction','rb'))
print("accuracy_score for spectogram "+str(accuracy_score(Spec_GroundTruth,Spec_Prediction)))
print("Class wise Precision,Recall Report ")
print(classification_report(Spec_GroundTruth,Spec_Prediction))