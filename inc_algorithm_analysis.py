import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

dataset  = pd.read_csv('inc.csv')
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
x = bow_vectorizer.fit_transform(dataset['Tweets'].values.astype('U'))
x=x.toarray()
y = dataset.iloc[:,1].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

#Logistic Regression
lreg = LogisticRegression()
lreg.fit(xtrain, ytrain)
lr_prediction = lreg.predict(xtest)
lr_cm = confusion_matrix(ytest,lr_prediction)
lr_correct = lr_cm.trace(offset=0,axis1=0,axis2=1,dtype=None,out=None)
lr_total = lr_cm.sum()
lr_accuracy = (lr_correct/lr_total)*100

#Naive-Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(xtrain, ytrain)
nb_prediction = nb_classifier.predict(xtest)
nb_cm = confusion_matrix(ytest,nb_prediction)
nb_correct = nb_cm.trace(offset=0,axis1=0,axis2=1,dtype=None,out=None)
nb_total = nb_cm.sum()
nb_accuracy = (nb_correct/nb_total)*100

#SVM
svc_classifier = SVC(kernel='linear')
svc_classifier.fit(xtrain,ytrain)
svc_prediction = svc_classifier.predict(xtest)
svc_cm = confusion_matrix(ytest,svc_prediction)
svc_correct = svc_cm.trace(offset=0,axis1=0,axis2=1,dtype=None,out=None)
svc_total = svc_cm.sum()
svc_accuracy = (svc_correct/svc_total)*100

print(lr_accuracy,nb_accuracy,svc_accuracy)


Algorithms = ('Logistic Regression', 'Naive Bayes', 'SVM')
y_pos = np.arange(len(Algorithms))
count = [lr_accuracy,nb_accuracy,svc_accuracy]
 
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, Algorithms)
plt.ylabel('Accuracy')
plt.title('Congress Tweet Analysis')
 
plt.show()




