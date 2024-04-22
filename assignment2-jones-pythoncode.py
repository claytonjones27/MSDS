#!/usr/bin/env python
# coding: utf-8

# In[72]:


import math
import pandas as pd
import numpy as np
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import GradientBoostingClassifier 


sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

df = pd.read_csv("HMEQ_loss.csv")


# In[73]:


dt = df.dtypes

objList = []
numList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A ]) : continue
    if dt[i] in (["object"]) : objList.append(i)
    if dt[i] in (["float64","int64"]) : numList.append(i)


"""
FILL IN MISSING WITH THE CATEGORY "MISSING"
"""
for i in objList :
    if df[i].isna().sum() == 0 : continue
    NAME = "IMP_"+i
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna("MISSING")
    g = df.groupby( NAME )
    df = df.drop(i, axis=1)

dt = df.dtypes
objList = []
for i in dt.index :
    if i in ([TARGET_F, TARGET_A]) : continue
    if dt[i] in (["object"]) : objList.append(i)


# In[74]:


#One hot encoding binary variables.
df["IMP_REASON_DebtCon_Yes"] = (df["IMP_REASON"].isin(["DebtCon"])+0)
df = df.drop("IMP_REASON", axis=1)


# In[75]:


#One hot encoding the categorical variable IMP_JOB.
dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]) : continue
    if dt[i] in (["object"]) : objList.append(i)

for i in objList:
    thePrefix = "z_" + i
    y = pd.get_dummies(df[i], prefix=thePrefix, dummy_na=False)   
    df = pd.concat([df, y], axis=1)
    df = df.drop(i, axis=1)


# In[76]:


#Imputing all missing numerical data and creating a flag variable to indicate if the value was missing.
for i in numList :
    if df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    df[FLAG] = df[i].isna() + 0
    df[IMP] = df[i]
    df.loc[df[IMP].isna(), IMP] = df[i].median()
    df = df.drop(i, axis=1)


# In[77]:


"""
SPLIT DATA
"""

X = df.copy()
X = X.drop(TARGET_F, axis=1)
X = X.drop(TARGET_A, axis=1)

Y = df[[TARGET_F, TARGET_A]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=1)

# print("FLAG DATA")
# print("TRAINING = ", X_train.shape)
# print("TEST = ", X_test.shape)


# In[78]:


F = ~ Y_train[TARGET_A].isna()
W_train = X_train[F].copy()
Z_train = Y_train[F].copy()

F = ~ Y_test[TARGET_A].isna()
W_test = X_test[F].copy()
Z_test = Y_test[F].copy()
print("TRAINING DATA")
print(Z_train.describe())
print("\nTEST DATA")
print(Z_test.describe())
print("\n\n")
#
##print(" ====== "")
##print("AMOUNT DATA")
##print("TRAINING = ", W_train.shape)
##print("TEST = ", Z_test.shape)

#Because the means were similar and the maxes were not too different, I am not going to take out any outliers.


# In[79]:


"""
DECISION TREE
"""

def getTreeVars(TREE, varNames) :
    tree_ = TREE.tree_
    varName = [varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    nameSet = set()
    for i in tree_.feature :
        if i != _tree.TREE_UNDEFINED :
            nameSet.add(i)
    nameList = list(nameSet)
    parameter_list = list()
    for i in nameList :
        parameter_list.append(varNames[i])
    return parameter_list


# In[80]:


#DEFAULT PROBABILITY

fm01_Tree = tree.DecisionTreeClassifier(max_depth=3)
# fm01_Tree = tree.DecisionTreeClassifier()
fm01_Tree = fm01_Tree.fit(X_train, Y_train[TARGET_F])

Y_Pred_train = fm01_Tree.predict(X_train)
Y_Pred_test = fm01_Tree.predict(X_test)


print("\n=============\n")
print("DECISION TREE\n")
print("Probability of default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")


# In[81]:


probs = fm01_Tree.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = fm01_Tree.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_tree = fpr_test
tpr_tree = tpr_test
auc_tree = roc_auc_test


plt.title('TREE ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red" )
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


feature_cols = list(X.columns.values)
tree.export_graphviz(fm01_Tree,out_file='tree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"])
vars_tree_flag = getTreeVars(fm01_Tree, feature_cols)
print("\nVariables Included in Decision Tree that Predict Loan Default:")
for i in vars_tree_flag:
    print(i)


# In[82]:


# DEFAULT AMOUNTS

amt_m01_Tree = tree.DecisionTreeRegressor(max_depth= 3)
amt_m01_Tree = amt_m01_Tree.fit(W_train, Z_train[TARGET_A])

Z_Pred_train = amt_m01_Tree.predict(W_train)
Z_Pred_test = amt_m01_Tree.predict(W_test)

# print("MEAN Train", Z_train[TARGET_A].mean())
# print("MEAN Test", Z_test[TARGET_A].mean())
# print("-------\n")

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("TREE RMSE Train:", RMSE_TRAIN )
print("TREE RMSE Test:", RMSE_TEST )

RMSE_TREE = RMSE_TEST

feature_cols = list(X.columns.values)
vars_tree_amt = getTreeVars( amt_m01_Tree, feature_cols ) 
tree.export_graphviz(amt_m01_Tree,out_file='tree_a.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, precision=0  )

print("\nVariables Included in Decision Tree that Predict Loss Amount:")
for i in vars_tree_amt :
   print(i)

# print("\n=======\n")
# print("FLAG=",vars_tree_flag)
# print("-------\n")
# print("AMOUNT=",vars_tree_amt)


# In[83]:


"""
RANDOM FOREST
"""



def getEnsembleTreeVars(ENSTREE, varNames) :
   importance = ENSTREE.feature_importances_
   index = np.argsort(importance)
   theList = []
   for i in index :
       imp_val = importance[i]
       if imp_val > np.average(ENSTREE.feature_importances_) :
           v = int(imp_val/np.max(ENSTREE.feature_importances_) * 100)
           theList.append((varNames[i], v))
   theList = sorted(theList,key=itemgetter(1),reverse=True)
   return theList


# In[84]:


# DEFAULT PROBABILITY

fm01_RF = RandomForestClassifier(n_estimators = 25, random_state=1)
fm01_RF = fm01_RF.fit(X_train, Y_train[TARGET_F])

Y_Pred_train = fm01_RF.predict(X_train)
Y_Pred_test = fm01_RF.predict(X_test)

print("\n=============\n")
print("RANDOM FOREST\n")
print("Probability of default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")


# In[85]:


probs = fm01_RF.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = fm01_RF.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_RF = fpr_test
tpr_RF = tpr_test
auc_RF = roc_auc_test

feature_cols = list(X.columns.values)
vars_RF_flag = getEnsembleTreeVars(fm01_RF, feature_cols)


plt.title('RF ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print("\nVariables Included in Decision Tree that Predict Loan Default:")
for i in vars_RF_flag :
   print(i)


# In[86]:


# DEFAULT AMOUNTS

amt_m01_RF = RandomForestRegressor(n_estimators = 25, random_state=1)
amt_m01_RF = amt_m01_RF.fit(W_train, Z_train[TARGET_A])

Z_Pred_train = amt_m01_RF.predict(W_train)
Z_Pred_test = amt_m01_RF.predict(W_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("RF RMSE Train:", RMSE_TRAIN )
print("RF RMSE Test:", RMSE_TEST )

RMSE_RF = RMSE_TEST

feature_cols = list( X.columns.values )
vars_RF_amt = getEnsembleTreeVars( amt_m01_RF, feature_cols )
print("\nVariables Included in Decision Tree that Predict Loss Amount:")
for i in vars_RF_amt :
    print(i)


# In[87]:


"""
GRADIENT BOOSTING
"""


fm01_GB = GradientBoostingClassifier(random_state=1)
fm01_GB = fm01_GB.fit(X_train, Y_train[ TARGET_F ])

Y_Pred_train = fm01_GB.predict(X_train)
Y_Pred_test = fm01_GB.predict(X_test)

print("\n=============\n")
print("GRADIENT BOOSTING\n")
print("Probability of crash")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")


# In[88]:




probs = fm01_GB.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = fm01_GB.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_GB = fpr_test
tpr_GB = tpr_test
auc_GB = roc_auc_test


feature_cols = list( X.columns.values )
vars_GB_flag = getEnsembleTreeVars( fm01_GB, feature_cols )


plt.title('GB ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print("\nVariables Included in Decision Tree that Predict Loan Default:")
for i in vars_GB_flag :
   print(i)



# In[89]:


# DEFAULT AMOUNTS

amt_m01_GB = GradientBoostingRegressor(random_state=1)
amt_m01_GB = amt_m01_GB.fit(W_train, Z_train[TARGET_A])

Z_Pred_train = amt_m01_GB.predict(W_train)
Z_Pred_test = amt_m01_GB.predict(W_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("GB RMSE Train:", RMSE_TRAIN)
print("GB RMSE Test:", RMSE_TEST)

RMSE_GB = RMSE_TEST

feature_cols = list( X.columns.values )
vars_GB_amt = getEnsembleTreeVars(amt_m01_GB, feature_cols)

print("\nVariables Included in Decision Tree that Predict Loss Amount:")
for i in vars_GB_amt :
   print(i)


# In[90]:


# COMPARING THE THREE MODELS

plt.title('MODELS ROC CURVE')
plt.plot(fpr_tree, tpr_tree, label = 'AUC TREE = %0.2f' % auc_tree, color="red")
plt.plot(fpr_RF, tpr_RF, label = 'AUC RF = %0.2f' % auc_RF, color="green")
plt.plot(fpr_GB, tpr_GB, label = 'AUC GB = %0.2f' % auc_GB, color="blue")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



print("Root Mean Square Average For Default Amount")
print("TREE", round(RMSE_TREE,2))
print("RF", round(RMSE_RF,2))
print("GB", round(RMSE_GB,2))

