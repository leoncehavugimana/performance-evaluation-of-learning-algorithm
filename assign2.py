import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.neural_network import MLPClassifier
my_data = np.array(pd.read_csv('D:\MLassign2\spambase.DATA',header=None)) #import data file in numpy array
X=my_data[:-1,:-1] #samples dataset
Y_data=my_data[:-1,-1:]
Y=np.reshape(Y_data,(4600)) #labels
np.set_printoptions(precision=3)
clf = svm.SVC()
scoring=['accuracy','f1']
cv=StratifiedKFold(n_splits=10)
scores=cross_validate(clf,X,Y,cv=cv,scoring=scoring)
print("----------------------SUPPORT VECTOR CLASSIFIER---------------------")#SVC code:
print(scores['fit_time'])
print("Training time(fit_time): Avg_ %0.2f  Stdev_ %0.2f \n" %(scores['fit_time'].mean(),scores['fit_time'].std()))
print(scores['test_accuracy'])
print("Test accuracy(test_accuracy): Avg_ %0.2f  Stdev_ %0.2f \n" %(scores['test_accuracy'].mean(),scores['test_accuracy'].std()))
print(scores['test_f1'])
print("Test f1(F1 measure): Avg_ %0.2f  Stdev_%0.2f \n" %(scores['test_f1'].mean(),scores['test_f1'].std()))

print("----------------------DECISION TREE CLASSIFIER----------------------") #DTC code:
clfTree = tree.DecisionTreeClassifier()
scoresTree=cross_validate(clfTree,X,Y,cv=cv,scoring=scoring)
print(scoresTree['fit_time'])
print("Training time(fit_time): Avg_ %0.2f  Stdev_ %0.2f \n" %(scoresTree['fit_time'].mean(),scoresTree['fit_time'].std()))
print(scoresTree['test_accuracy'])
print("Test accuracy(test_accuracy): Avg_ %0.2f  Stdev_ %0.2f \n" %(scoresTree['test_accuracy'].mean(),scoresTree['test_accuracy'].std()))
print(scoresTree['test_f1'])
print("Test f1(F1 measure): Avg_ %0.2f  Stdev_%0.2f \n" %(scoresTree['test_f1'].mean(),scoresTree['test_f1'].std()))

print("----------------------MLP CLASSIFIER----------------------") #MLP code
clfMLP = MLPClassifier()
scoresMLP=cross_validate(clfMLP,X,Y,cv=cv,scoring=scoring)
print(scoresMLP['fit_time'])
print("Training time(fit_time): Avg_ %0.2f  Stdev_ %0.2f \n" %(scoresMLP['fit_time'].mean(),scoresMLP['fit_time'].std()))
print(scoresMLP['test_accuracy'])
print("Test accuracy(test_accuracy): Avg_ %0.2f  Stdev_ %0.2f \n" %(scoresMLP['test_accuracy'].mean(),scoresMLP['test_accuracy'].std()))
print(scoresMLP['test_f1'])
print("Test f1(F1 measure): Avg_ %0.2f  Stdev_%0.2f \n" %(scoresMLP['test_f1'].mean(),scoresMLP['test_f1'].std()))










