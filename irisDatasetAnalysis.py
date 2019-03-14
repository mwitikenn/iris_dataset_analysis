#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
import seaborn as sns


# In[2]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[3]:


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# In[4]:


# shape
print(dataset.shape)


# In[5]:


# head
print(dataset.head(20))


# In[6]:


# descriptions
print(dataset.describe())


# In[8]:


# mean
print(dataset.mean())


# In[10]:


# max
print(dataset.max())


# In[11]:


# standard deviation
print(dataset.std())


# In[14]:


# count
print(dataset.count())


# In[15]:


# class distribution
print(dataset.groupby('class').size())


# In[20]:


# box and whisker plots
print(dataset.median())
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[17]:


# histograms
dataset.hist()
plt.show()


# In[18]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# In[23]:


sns.heatmap(dataset.corr(),annot = True,cmap='RdYlGn',linewidths=3.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[30]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
print(X)
Y = array[:,4]
print(Y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[32]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
print(X_train)


# In[33]:


print(Y_train)


# In[34]:


print(X_validation)


# In[35]:


print(Y_validation)


# In[36]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[37]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[38]:


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




