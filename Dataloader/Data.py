import os
import pandas as pd
from sklearn.model_selection import train_test_split
monkeypox = os.listdir("/home/sci/mkaranam/Desktop/CS6190---ProbabilisticMLProject/OriginalImages/OriginalImages/Monkey Pox")
not_monkey_pox = os.listdir("/home/sci/mkaranam/Desktop/CS6190---ProbabilisticMLProject/OriginalImages/OriginalImages/Others")
labels = [1]*len(monkeypox)+[0]*len(not_monkey_pox)
dataset_  = pd.DataFrame()
dataset_["Image_name"] = monkeypox+not_monkey_pox
dataset_['labels'] = labels
check = list(dataset_["Image_name"][:len(monkeypox)])
not_check = list(dataset_["Image_name"][len(monkeypox):])
check_labels = list(dataset_["labels"][:len(monkeypox)])
not_check_labels = list(dataset_["labels"][len(monkeypox):])
dataset_ = dataset_.sample(frac=1).reset_index(drop=True)
dataset_.to_csv("MonkeypoxDataset.csv",index=False)
data = pd.read_csv("/home/sci/mkaranam/Desktop/CS6190---ProbabilisticMLProject/data/MonkeypoxDataset.csv")
Y = data['labels']
# X = data.drop('labels',axis=1)
X=  data
X_train,X_cv,Y_train, Y_cv = train_test_split(X,Y,test_size=0.15,stratify=Y,random_state=33)
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.2,stratify=Y_train,random_state=33)
X_train.to_csv("trainMonkeypox.csv", index=False)
X_cv.to_csv("cvMonkeypox.csv", index=False)
X_test.to_csv("testMonkeypox.csv",index=False)