# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:55:12 2020

@author: Rashmi, B19218
Ph no. : 7015331137
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score  
from sklearn.neighbors import KNeighborsClassifier 

data = pd.read_csv('seismic_bumps1.csv')
     
k_value =[1,3,5]        
remove_attributes = ['nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89']
data.drop(columns=remove_attributes, inplace=True)         
        
data_c0 = data[data['class'] == 0]  # 'class 0' Data
data_c1 = data[data['class'] == 1]  # 'class 1' Data         
        
        
x0 = data_c0.drop(columns=['class'])    # droping 'class' attribute 
x0_label = data_c0['class']             # storing 'class' 

x1 = data_c1.drop(columns=['class'])
x1_label = data_c1['class']        
        
# Following will split data into training and testing datasets with 70% and 30% respectively
x0_train, x0_test, x0_label_train, x0_label_test = train_test_split(x0, x0_label, test_size=0.3,random_state=42, shuffle=True) 
x1_train, x1_test, x1_label_train, x1_label_test = train_test_split(x1, x1_label, test_size=0.3,random_state=42, shuffle=True)

x_train, x_test, x_label_train, x_label_test = x0_train.append(x1_train), x0_test.append(x1_test), x0_label_train.append(x1_label_train), x0_label_test.append(x1_label_test)  # combining both test and training samples of both example

x_train.merge(x_label_train, left_index=True, right_index=True).to_csv('seismic-bumps-train.csv', index=False) 
x_test.merge(x_label_test, left_index=True, right_index=True).to_csv('seismic-bumps-test.csv', index=False)
       
accuracy = {'KNN': 0, 'KNN-normalized': 0, 'Bayes': 0}

# Q1._______________________________________________________________
print('\n1.)')        
for k in k_value: 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, x_label_train)     # fitting the model
    x_label_pred = knn.predict(x_test)  # pridicting 

    accuracy['KNN'] = max(  accuracy['KNN'],accuracy_score(x_label_test,x_label_pred) * 100) # calculate accuracy
    
    print('For K = {}:'.format(k))
    print('Confusion matrix =>\n', confusion_matrix(x_label_test, x_label_pred))   # gives confusion matrix
    print('\nClassification Accuracy = {:.3f}\n'.format(accuracy_score(x_label_test, x_label_pred)*100))  # gives classification accuracy
        
# Q2.___________________________________________________________________
print('\n2.)')        
x_train_normalised = (x_train - x_train.min()) / (x_train.max() - x_train.min())  #normalizing train-data
x_test_normalised = (x_test - x_train.min()) / (x_train.max() - x_train.min())  #normalizing test-data using min-max of train-data

x_train_normalised.merge(x_label_train, left_index=True, right_index=True).to_csv('seismic-bumps-train-Normalised.csv', index=False)   # mereging to make save csv file of training data
x_test_normalised.merge(x_label_test, left_index=True, right_index=True).to_csv('seismic-bumps-test-Normalised.csv', index=False)  # merging test data to save as csv

for k in k_value:
    knn = KNeighborsClassifier(n_neighbors=k)   # classifying data
    knn.fit(x_train_normalised, x_label_train)   # fitting curve
    x_label_pred = knn.predict(x_test_normalised)
   
    accuracy['KNN-normalized'] = max(accuracy['KNN-normalized'],accuracy_score(x_label_test, x_label_pred) * 100) # saving maximum accuracy of knn-normalized
    print('For K = {}:'.format(k))
    print('Confusion matrix =>\n', confusion_matrix(x_label_test, x_label_pred)) #printing confusion matrix
    print('\nClassification Accuracy = {:.3f}\n'.format(accuracy_score(x_label_test, x_label_pred)*100)) #printing accuracy of knn
    
# Q3.__________________________
print('\n3.)')            
def fun(a,b): 
    c = [value for value in a if value in b] 
    return len(c)

def maxLiklihood(m, covar , x):     # function to calcuate maximum likelihood
    determinent = np.linalg.det(covar)    # determinent of covariance matrix
    val = 1/(((2*np.pi)*5.)*determinent*0.5)    
    signmainv = np.linalg.inv(covar);    # invariance of covariance matrix
    firstparmeter = np.transpose(x-m);                   
    firstparmeter = np.dot(firstparmeter,signmainv)
    
    finalparmeter = np.dot(firstparmeter,(x-m))  # mahalnobis distance
    finalparmeter = -finalparmeter/2;
    ans = val*np.exp(finalparmeter);   
    return ans
        
mean_c0 = x0_train.mean()  #calculating mean of class0
mean_c1 = x1_train.mean()  # calcuating mean of class1
covar_c0 = np.cov(x0_train.T) # calculating covariance class0
covar_c1 = np.cov(x1_train.T)  #calculating covariance of class1

c1_predicted = []
c0_predicted = []
c1_actual = []
c0_actual = []

for i in x_test.index:
   
    m0 = maxLiklihood(mean_c0,covar_c0,x_test.loc[i])  # maximum likelihood with mean and covariance vector as class0
    m1 = maxLiklihood(mean_c1,covar_c1,x_test.loc[i])  # maximum likelihood with mean and covariance vector as class1
    m0 = m0*x0_train.size
    m1 = m1*x1_train.size
    if(m0>m1):
        c0_predicted.append(i)  # append class according to maximum likelihood value
    else:
        c1_predicted.append(i)
for index,items in x_label_test.iteritems():
    if items==1:
        c1_actual.append(index)  # append indics with actual class1
    else:
        c0_actual.append(index) # append indics with actual class0
    

c0_correct =   fun(c0_actual,c0_predicted)    # no. of samples whose class were correctly predicted as 0
c1_correct =   fun(c1_actual,c1_predicted)    # whose class were correctly predicted as 1
Class1Class0  =   len(c1_actual)-c1_correct   # whose class were wrongly predicted as 0
Class0Class1  =   len(c0_actual)-c0_correct   # whose class were wrongly predicted as 1
matrix = np.array([[c0_correct,Class0Class1],[Class1Class0,c1_correct]]) # gives confusion matrix

Accuracy = (matrix[0][0] + matrix[1][1])/(matrix[1][1] + matrix[0][1] + matrix[1][0] + matrix[0][0]) # calculating accuracy
print(matrix)
print('\nAccuracy = {:.3f}\n'.format(Accuracy))
accuracy['Bayes'] = max(Accuracy*100 , 0)

# Q4._____________________________
print('\n4.)')
print("Maximum accuracy for all three methods:")
for i in accuracy.keys():
    print(i,end = " ")
    print('=> {:.3f}\n'.format(accuracy[i]))  #gives maximum accuracy for all methods        
        
               