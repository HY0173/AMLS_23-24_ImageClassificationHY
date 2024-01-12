import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# ==================== Pretrained Models + SVM/KNN/DecisionTree Classifier ====================

def Train_classifier(feature_extractor,clf,device,X_train,Y_train,X_test,Y_test):
    # Use Pretrained models to Extract features
    feature_extractor = feature_extractor.to(device)
    X_train = X_train.to(device)
    features = feature_extractor(X_train)
    
    # Define Classifier
    if clf == "SVM":
        classifier = svm.LinearSVC()
    elif clf == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif clf == "Tree":
        classifier = DecisionTreeClassifier()

    features = features.detach().cpu().numpy()
    Y_train = Y_train.detach().cpu().numpy()
   
    # Training
    print("----------------- Training {} Classifier -----------------".format(clf))
    classifier.fit(features,Y_train)

    # Testing
    X_test = X_test.to(device)
    feature_test = feature_extractor(X_test)
    feature_test = feature_test.detach().cpu().numpy()
    result = classifier.predict(feature_test)
    Y_test = Y_test.detach().cpu().numpy()
    
    # Accuracy,Precision,Recall,F1_score
    confusion_m = metrics.confusion_matrix(Y_test,result)
    acc = metrics.accuracy_score(Y_test,result)
    print("Accuracy: ",acc)
    pre = metrics.precision_score(Y_test,result)
    print("Precision: ",pre)
    recall = metrics.recall_score(Y_test,result)
    print("Recall: ",recall)
    f1 = metrics.f1_score(Y_test,result)
    print("F1 score: ",f1)
    auc = metrics.accuracy_score(Y_test,result)
    print("AUC: ",auc)
    return classifier

# Get X&Y
def get_data(dataset):
    X,Y = [],[]
    for idx,data in tqdm(enumerate(dataset)):
        x,y = data[0],data[1]
        x = torch.cat((x,x,x), 0).tolist()
        y = torch.tensor(y).tolist()
        X.append(x)
        Y.append(y)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y