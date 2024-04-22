#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import warnings
warnings.filterwarnings("ignore")



sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

df = pd.read_csv("HMEQ_loss.csv")


# In[2]:


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


# In[3]:


#One hot encoding binary variables.
df["IMP_REASON_DebtCon_Yes"] = (df["IMP_REASON"].isin(["DebtCon"])+0)
df = df.drop("IMP_REASON", axis=1)


# In[4]:


dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]) : continue
    if dt[i] in (["object"]) : objList.append(i)

for i in objList:
    thePrefix = "z_" + i
    y = pd.get_dummies(df[i], prefix=thePrefix, dummy_na=False)   
    df = pd.concat([df, y], axis=1)
    #df = df.drop(i, axis=1)



# In[5]:


#Imputing all missing numerical data and creating a flag variable to indicate if the value was missing.
for i in numList :
    if df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    df[FLAG] = df[i].isna() + 0
    df[IMP] = df[i]
    df.loc[df[IMP].isna(), IMP] = df[i].median()
    df = df.drop(i, axis=1)


# In[6]:


"""
Remove Outliers
"""


dt = df.dtypes
numList = []
for i in dt.index:
    #print(i, dt[i])
    if i in ([TARGET_F, TARGET_A ]) : continue
    if dt[i] in (["float64","int64"]) : numList.append(i)

# for i in numList:
#     print(i)


# In[7]:


for i in numList:
    theMean = df[i].mean()
    theSD = df[i].std()
    theMax = df[i].max()
    theCutoff = round(theMean + 3*theSD)
    if theMax < theCutoff : continue
    FLAG = "O_" + i
    TRUNC = "TRUNC_" + i
    df[FLAG] = (df[i] > theCutoff)+ 0
    df[TRUNC] = df[ i ]
    df.loc[ df[TRUNC] > theCutoff, TRUNC] = theCutoff
    df = df.drop(i, axis=1 )


# In[8]:


for i in objList:
    df = df.drop(i, axis=1)


# In[9]:


"""
SPLIT DATA
"""

X = df.copy()
X = X.drop(TARGET_F, axis=1)
X = X.drop(TARGET_A, axis=1)

Y = df[[TARGET_F, TARGET_A]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=2)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=2)
# print("FLAG DATA")
# print("TRAINING = ", X_train.shape)
# print("TEST = ", X_test.shape)


# In[10]:


F = ~ Y_train[TARGET_A].isna()
W_train = X_train[F].copy()
Z_train = Y_train[F].copy()

F = ~ Y_test[TARGET_A].isna()
W_test = X_test[F].copy()
Z_test = Y_test[F].copy()


# In[11]:


"""
MODEL ACCURACY METRICS
"""

def getProbAccuracyScores(NAME, MODEL, X, Y) :
    pred = MODEL.predict(X)
    probs = MODEL.predict_proba(X)
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

def print_ROC_Curve(TITLE, LIST) :
    fig = plt.figure(figsize=(6,4))
    plt.title( TITLE )
    for theResults in LIST :
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot(fpr, tpr, label = theLabel)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def print_Accuracy(TITLE, LIST) :
    print( TITLE )
    print( "======" )
    for theResults in LIST :
        NAME = theResults[0]
        ACC = theResults[1]
        print( NAME, " = ", ACC )
    print( "------\n\n" )

def getAmtAccuracyScores(NAME, MODEL, X, Y) :
    pred = MODEL.predict(X)
    MEAN = Y.mean()
    RMSE = math.sqrt( metrics.mean_squared_error(Y, pred))
    return [NAME, RMSE, MEAN]


# In[12]:


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


# In[13]:


#DEFAULT PROBABILITY

WHO = "TREE"

DFT = tree.DecisionTreeClassifier(max_depth=3)
# fm01_Tree = tree.DecisionTreeClassifier()
DFT = DFT.fit(X_train, Y_train[TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train, Y_train[TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test, Y_test[TARGET_F])

# print_ROC_Curve(WHO,[TRAIN_DFT, TEST_DFT]) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])

feature_cols = list(X.columns.values)
tree.export_graphviz(DFT,out_file='tree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"])
vars_tree_flag = getTreeVars(DFT, feature_cols)


# In[14]:


# DEFAULT AMOUNTS

AMT = tree.DecisionTreeRegressor(max_depth= 3)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A])
# print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT ])

feature_cols = list(X.columns.values)
vars_tree_amt = getTreeVars(AMT, feature_cols) 
tree.export_graphviz(AMT,out_file='tree_a.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, precision=0  )


# In[15]:


TREE_DFT = TEST_DFT.copy()
TREE_AMT = TEST_AMT.copy()


# In[16]:


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


# In[17]:


# DEFAULT PROBABILITY

WHO = "RF"

DFT = RandomForestClassifier(n_estimators = 25, random_state=1)
DFT = DFT.fit(X_train, Y_train[TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train, Y_train[TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test, Y_test[TARGET_F])

# print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


feature_cols = list(X.columns.values )
vars_RF_flag = getEnsembleTreeVars(DFT, feature_cols)


# In[18]:


# DEFAULT AMOUNTS

AMT = RandomForestRegressor(n_estimators = 25, random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A])
# print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT ])

feature_cols = list(X.columns.values)
vars_RF_amt = getEnsembleTreeVars(AMT, feature_cols)


# In[19]:


RF_DFT = TEST_DFT.copy()
RF_AMT = TEST_AMT.copy()


# In[20]:


"""
GRADIENT BOOSTING
"""
WHO = "GB"

DFT = GradientBoostingClassifier(random_state=1)
DFT = DFT.fit(X_train, Y_train[TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train, Y_train[ TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test, Y_test[TARGET_F])

# print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


feature_cols = list(X.columns.values )
vars_GB_flag = getEnsembleTreeVars(DFT, feature_cols)


# In[21]:


# DEFAULT AMOUNTS

AMT = GradientBoostingRegressor(random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A])
# print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT ])

feature_cols = list(X.columns.values)
vars_GB_amt = getEnsembleTreeVars(AMT, feature_cols) 


# In[22]:


GB_DFT = TEST_DFT.copy()
GB_AMT = TEST_AMT.copy()


# In[23]:


def getCoefLogit(MODEL, TRAIN_DATA) :
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_[0]
    for coef, feat in zip(MODEL.coef_[0],varNames):
        coef_dict[feat] = coef
    print("\nDEFAULT")
    print("---------")
    print("Total Variables: ", len(coef_dict))
    for i in coef_dict:
        print(i, " = ", coef_dict[i])



def getCoefLinear(MODEL, TRAIN_DATA) :
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_
    for coef, feat in zip(MODEL.coef_,varNames):
        coef_dict[feat] = coef
    print("\nDEFAULT AMOUNT")
    print("---------")
    print("Total Variables: ", len(coef_dict))
    for i in coef_dict :
        print(i, " = ", coef_dict[i])


# In[24]:


"""
REGRESSION ALL VARIABLES
"""

WHO = "REG_ALL"

DFT = LogisticRegression(solver='newton-cg', max_iter=1000)
DFT = DFT.fit(X_train, Y_train[TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train, Y_train[TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test, Y_test[TARGET_F])

# print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT]) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT] )


# In[25]:


# DEFAULT AMOUNTS

AMT = LinearRegression()
AMT = AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A] )
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])


varNames = list(X_train.columns.values)

# REG_ALL_DFT_COEF = getCoefLogit(DFT, X_train)
# REG_ALL_AMT_COEF = getCoefLinear(AMT, X_train)


# In[26]:


REG_ALL_DFT = TEST_DFT.copy()
REG_ALL_AMT = TEST_AMT.copy()


# In[27]:


"""
REGRESSION DECISION TREE
"""

WHO = "REG_TREE"

DFT = LogisticRegression( solver='newton-cg', max_iter=1000)
DFT = DFT.fit(X_train[vars_tree_flag], Y_train[ TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train[vars_tree_flag], Y_train[TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test[vars_tree_flag], Y_test[TARGET_F])

# print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT]) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY",[TRAIN_DFT, TEST_DFT])


# In[28]:


# DEFAULT AMOUNTS

AMT = LinearRegression()
AMT = AMT.fit(W_train[vars_tree_amt], Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[vars_tree_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test[vars_tree_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])


varNames = list(X_train.columns.values)

#REG_TREE_DFT_COEF = getCoefLogit(DFT, X_train[vars_tree_flag])
#REG_TREE_AMT_COEF = getCoefLinear(AMT, X_train[vars_tree_amt])


# In[29]:


REG_TREE_DFT = TEST_DFT.copy()
REG_TREE_AMT = TEST_AMT.copy()


# In[30]:


"""
REGRESSION RANDOM FOREST
"""

WHO = "REG_RF"


print("\n\n")
RF_flag = []
for i in vars_RF_flag:
    print(i)
    theVar = i[0]
    RF_flag.append(theVar)

print("\n\n")
RF_amt = []
for i in vars_RF_amt:
    print(i)
    theVar = i[0]
    RF_amt.append(theVar)


# In[31]:


DFT = LogisticRegression(solver='newton-cg', max_iter=1000)
DFT = DFT.fit(X_train[RF_flag], Y_train[TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train[RF_flag], Y_train[TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test[RF_flag], Y_test[TARGET_F])

# print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[32]:


# DEFAULT AMOUNTS

AMT = LinearRegression()
AMT = AMT.fit(W_train[RF_amt], Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[RF_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test[RF_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ])


REG_RF_DFT_COEF = getCoefLogit(DFT, X_train[RF_flag])
REG_RF_AMT_COEF = getCoefLinear(AMT, X_train[RF_amt])


# In[33]:


REG_RF_DFT = TEST_DFT.copy()
REG_RF_AMT = TEST_AMT.copy()


# In[34]:


"""
REGRESSION GRADIENT BOOSTING
"""

WHO = "REG_GB"


print("\n\n")
GB_flag = []
for i in vars_GB_flag:
    print(i)
    theVar = i[0]
    GB_flag.append(theVar)

print("\n\n")
GB_amt = []
for i in vars_GB_amt:
    print(i)
    theVar = i[0]
    GB_amt.append(theVar)


# In[35]:


DFT = LogisticRegression(solver='newton-cg', max_iter=1000)
DFT = DFT.fit(X_train[GB_flag], Y_train[TARGET_F])

TRAIN_DFT = getProbAccuracyScores(WHO + "_Train", DFT, X_train[GB_flag], Y_train[TARGET_F])
TEST_DFT = getProbAccuracyScores(WHO, DFT, X_test[GB_flag], Y_test[TARGET_F])

# print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
# print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[36]:


# DEFAULT AMOUNTS

AMT = LinearRegression()
AMT = AMT.fit( W_train[GB_amt], Z_train[TARGET_A] )

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

REG_GB_DFT_COEF = getCoefLogit(DFT, X_train[GB_flag])
REG_GB_AMT_COEF = getCoefLinear(AMT, X_train[GB_amt])


# In[37]:


REG_GB_DFT = TEST_DFT.copy()
REG_GB_AMT = TEST_AMT.copy()


# In[38]:


import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
theScaler = MinMaxScaler()
theScaler.fit(X_train)


# In[39]:


def get_TF_ProbAccuracyScores(NAME, MODEL, X, Y) :
    probs = MODEL.predict(X)
    pred_list = []
    for p in probs :
        pred_list.append(np.argmax(p))
    pred = np.array(pred_list)
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]


# In[40]:


WHO = "Tensor_Flow_RELU_One_Layer"

U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_train.columns.values)

U_train = U_train[GB_flag]
U_test = U_test[GB_flag]


# In[41]:


F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.relu
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)


# In[42]:


F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
#F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
#F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)


# In[43]:


DFT = tf.keras.Sequential()
DFT.add(F_LAYER_01)
# DFT.add(F_LAYER_DROP)
# DFT.add(F_LAYER_02)
DFT.add(F_LAYER_OUTPUT)
DFT.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
DFT.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)


# In[44]:


TRAIN_DFT = get_TF_ProbAccuracyScores(WHO + "_Train", DFT, U_train, Y_train[TARGET_F])
TEST_DFT = get_TF_ProbAccuracyScores(WHO, DFT, U_test, Y_test[TARGET_F])

print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[45]:


V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_train.columns.values)

V_train = V_train[GB_amt]
V_test = V_test[GB_amt]


# In[46]:


A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.relu
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int(2*A_theShapeSize)


# In[47]:


A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
# A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
# A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)


# In[48]:


AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
# AMT.add(A_LAYER_DROP)
# AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)


# In[49]:


TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

TF_DFT_RELU_One_Layer = TEST_DFT.copy()
TF_AMT_RELU_One_Layer = TEST_AMT.copy()


# In[50]:


ALL_DFT = [TREE_DFT, RF_DFT, GB_DFT, REG_ALL_DFT, REG_TREE_DFT, REG_RF_DFT, REG_GB_DFT, TF_DFT_RELU_One_Layer]

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_DFT) 

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_DFT)


# In[51]:


ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_RELU_One_Layer]
ALL_AMT = sorted(ALL_AMT, key = lambda x: x[1])
print_Accuracy("ALL DAMAGE MODEL ACCURACY", ALL_AMT)


# In[52]:


# RELU Two Layers and Dropout Layer
WHO = "Tensor_Flow_RELU_Two_Layers"

U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_train.columns.values)

U_train = U_train[GB_flag]
U_test = U_test[GB_flag]

F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.relu
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)

F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

DFT = tf.keras.Sequential()
DFT.add(F_LAYER_01)
DFT.add(F_LAYER_DROP)
DFT.add(F_LAYER_02)
DFT.add(F_LAYER_OUTPUT)
DFT.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
DFT.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

TRAIN_DFT = get_TF_ProbAccuracyScores(WHO + "_Train", DFT, U_train, Y_train[TARGET_F])
TEST_DFT = get_TF_ProbAccuracyScores(WHO, DFT, U_test, Y_test[TARGET_F])

print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[53]:


V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_train.columns.values)

V_train = V_train[GB_amt]
V_test = V_test[GB_amt]

A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.relu
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int(2*A_theShapeSize)

A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_DROP)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

TF_DFT_RELU_Two_Layers = TEST_DFT.copy()
TF_AMT_RELU_Two_Layers = TEST_AMT.copy()

ALL_DFT = [TREE_DFT, RF_DFT, GB_DFT, REG_ALL_DFT, REG_TREE_DFT, REG_RF_DFT, REG_GB_DFT, TF_DFT_RELU_Two_Layers]

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_DFT) 

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_DFT)


# In[54]:


ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_RELU_Two_Layers]
ALL_AMT = sorted(ALL_AMT, key = lambda x: x[1])
print_Accuracy("ALL DAMAGE MODEL ACCURACY", ALL_AMT)


# In[55]:


# Sigmoid Activation Function One Layer
WHO = "Tensor_Flow_Sigmoid_One_Layer"

U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_train.columns.values)

U_train = U_train[GB_flag]
U_test = U_test[GB_flag]

F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.sigmoid
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)

F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
# F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
# F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

DFT = tf.keras.Sequential()
DFT.add(F_LAYER_01)
# DFT.add(F_LAYER_DROP)
# DFT.add(F_LAYER_02)
DFT.add(F_LAYER_OUTPUT)
DFT.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
DFT.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

TRAIN_DFT = get_TF_ProbAccuracyScores(WHO + "_Train", DFT, U_train, Y_train[TARGET_F])
TEST_DFT = get_TF_ProbAccuracyScores(WHO, DFT, U_test, Y_test[TARGET_F])

print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[56]:


V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_train.columns.values)

V_train = V_train[GB_amt]
V_test = V_test[GB_amt]

A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.relu
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int(2*A_theShapeSize)

A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
# A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
# A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
# AMT.add(A_LAYER_DROP)
# AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

TF_DFT_Sigmoid_One_Layer = TEST_DFT.copy()
TF_AMT_Sigmoid_One_Layer = TEST_AMT.copy()

ALL_DFT = [TREE_DFT, RF_DFT, GB_DFT, REG_ALL_DFT, REG_TREE_DFT, REG_RF_DFT, REG_GB_DFT, TF_DFT_Sigmoid_One_Layer]

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_DFT) 

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_DFT)


# In[57]:


ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_Sigmoid_One_Layer]
ALL_AMT = sorted(ALL_AMT, key = lambda x: x[1])
print_Accuracy("ALL DAMAGE MODEL ACCURACY", ALL_AMT)


# In[58]:


# Sigmoid Activation Function Two Layers
WHO = "Tensor_Flow_Sigmoid_Two_Layers"

U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_train.columns.values)

U_train = U_train[GB_flag]
U_test = U_test[GB_flag]

F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.sigmoid
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)

F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

DFT = tf.keras.Sequential()
DFT.add(F_LAYER_01)
DFT.add(F_LAYER_DROP)
DFT.add(F_LAYER_02)
DFT.add(F_LAYER_OUTPUT)
DFT.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
DFT.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

TRAIN_DFT = get_TF_ProbAccuracyScores(WHO + "_Train", DFT, U_train, Y_train[TARGET_F])
TEST_DFT = get_TF_ProbAccuracyScores(WHO, DFT, U_test, Y_test[TARGET_F])

print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[60]:


V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_train.columns.values)

V_train = V_train[GB_amt]
V_test = V_test[GB_amt]

A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.relu
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int(2*A_theShapeSize)

A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_DROP)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

TF_DFT_Sigmoid_Two_Layers = TEST_DFT.copy()
TF_AMT_Sigmoid_Two_Layers = TEST_AMT.copy()

ALL_DFT = [TREE_DFT, RF_DFT, GB_DFT, REG_ALL_DFT, REG_TREE_DFT, REG_RF_DFT, REG_GB_DFT, TF_DFT_Sigmoid_Two_Layers]

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_DFT) 

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_DFT)


# In[61]:


ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_Sigmoid_Two_Layers]
ALL_AMT = sorted(ALL_AMT, key = lambda x: x[1])
print_Accuracy("ALL DAMAGE MODEL ACCURACY", ALL_AMT)


# In[62]:


# Tanh Activation Function One Layer
WHO = "Tensor_Flow_Tanh_One_Layer"

U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_train.columns.values)

U_train = U_train[GB_flag]
U_test = U_test[GB_flag]

F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.tanh
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)

F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
# F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
# F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

DFT = tf.keras.Sequential()
DFT.add(F_LAYER_01)
# DFT.add(F_LAYER_DROP)
# DFT.add(F_LAYER_02)
DFT.add(F_LAYER_OUTPUT)
DFT.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
DFT.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

TRAIN_DFT = get_TF_ProbAccuracyScores(WHO + "_Train", DFT, U_train, Y_train[TARGET_F])
TEST_DFT = get_TF_ProbAccuracyScores(WHO, DFT, U_test, Y_test[TARGET_F])

print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[63]:


V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_train.columns.values)

V_train = V_train[GB_amt]
V_test = V_test[GB_amt]

A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.relu
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int(2*A_theShapeSize)

A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
# A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
# A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
# AMT.add(A_LAYER_DROP)
# AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

TF_DFT_Tanh_One_Layer = TEST_DFT.copy()
TF_AMT_Tanh_One_Layer = TEST_AMT.copy()

ALL_DFT = [TREE_DFT, RF_DFT, GB_DFT, REG_ALL_DFT, REG_TREE_DFT, REG_RF_DFT, REG_GB_DFT, TF_DFT_Tanh_One_Layer]

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_DFT) 

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_DFT)


# In[64]:


ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_Tanh_One_Layer]
ALL_AMT = sorted(ALL_AMT, key = lambda x: x[1])
print_Accuracy("ALL DAMAGE MODEL ACCURACY", ALL_AMT)


# In[65]:


# Tanh Activation Function Two Layers
WHO = "Tensor_Flow_Tanh_Two_Layers"

U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_train.columns.values)

U_train = U_train[GB_flag]
U_test = U_test[GB_flag]

F_theShapeSize = U_train.shape[1]
F_theActivation = tf.keras.activations.sigmoid
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)

F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

DFT = tf.keras.Sequential()
DFT.add(F_LAYER_01)
DFT.add(F_LAYER_DROP)
DFT.add(F_LAYER_02)
DFT.add(F_LAYER_OUTPUT)
DFT.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
DFT.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

TRAIN_DFT = get_TF_ProbAccuracyScores(WHO + "_Train", DFT, U_train, Y_train[TARGET_F])
TEST_DFT = get_TF_ProbAccuracyScores(WHO, DFT, U_test, Y_test[TARGET_F])

print_ROC_Curve(WHO, [TRAIN_DFT, TEST_DFT] ) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_DFT, TEST_DFT])


# In[66]:


V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_train.columns.values)

V_train = V_train[GB_amt]
V_test = V_test[GB_amt]

A_theShapeSize = V_train.shape[1]
A_theActivation = tf.keras.activations.relu
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()
A_theEpochs = 800

A_theUnits = int(2*A_theShapeSize)

A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_DROP)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

TF_DFT_Tanh_Two_Layers = TEST_DFT.copy()
TF_AMT_Tanh_Two_Layers = TEST_AMT.copy()

ALL_DFT = [TREE_DFT, RF_DFT, GB_DFT, REG_ALL_DFT, REG_TREE_DFT, REG_RF_DFT, REG_GB_DFT, TF_DFT_Tanh_Two_Layers]

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_DFT) 

ALL_DFT = sorted(ALL_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_DFT)


# In[67]:


ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_Tanh_Two_Layers]
ALL_AMT = sorted(ALL_AMT, key = lambda x: x[1])
print_Accuracy("ALL DAMAGE MODEL ACCURACY", ALL_AMT)


# In[68]:


#Comparing All Tensor Flow Models


# In[69]:


ALL_TF_DFT = [TF_DFT_RELU_One_Layer, TF_DFT_RELU_Two_Layers,TF_DFT_Sigmoid_One_Layer,TF_DFT_Sigmoid_Two_Layers, TF_DFT_Tanh_One_Layer, TF_DFT_Tanh_Two_Layers]

ALL_TF_DFT = sorted(ALL_TF_DFT, key = lambda x: x[4], reverse=True)
print_ROC_Curve("ALL TF MODELS", ALL_TF_DFT) 

ALL_TF_DFT = sorted(ALL_TF_DFT, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL TF CLASSIFICATION ACCURACY", ALL_TF_DFT)


# In[70]:


ALL_TF_AMT = [TF_AMT_RELU_One_Layer, TF_AMT_RELU_Two_Layers,TF_AMT_Sigmoid_One_Layer,TF_AMT_Sigmoid_Two_Layers, TF_AMT_Tanh_One_Layer, TF_AMT_Tanh_Two_Layers]
ALL_TF_AMT = sorted(ALL_TF_AMT, key = lambda x: x[1])
print_Accuracy("ALL TF DAMAGE MODEL ACCURACY", ALL_TF_AMT)


# In[ ]:




