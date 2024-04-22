
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"



df = pd.read_csv("HMEQ_loss.csv")


print(df.head(5).T)



print("===============\n\n")
print("Data Types:\n")
print(df.dtypes)




#Descriptive statistics for the data.
print("\n===============\n")
print("Descriptive Statistics:")
print(df.describe().T)
print("\n")



#Creating lists for each variable type.
dt = df.dtypes
objList = []
intList = []
floatList = []
numList = []
for i in dt.index :
    if i in ([TARGET_F, TARGET_A]) : continue
    if dt[i] in (["object"]) : objList.append(i)
    if dt[i] in (["float64"]) : floatList.append(i)
    if dt[i] in (["int64"]) : intList.append(i)
    if dt[i] in (["int64","float64"]) : numList.append(i)



#Exploring statistics for object variables.
print("===============\n")
print("EDA:\n\n")
print("CATEGORICAL VARIABLES\n\n")
for i in objList :
    print("Class = ", i)
    g = df.groupby(i)
    x = g[TARGET_F].mean()
    print("Default Prob", x)
    print(".................")
    x = g[TARGET_A].mean()
    print( "Loss Amount", x )
    print("===============\n\n")



#Exploring statistics for integer variables.
print("===============\n")
print("INTEGER VARIABLES")
print("\n")
for i in intList :
    print("Variable=",i)
    g = df.groupby(TARGET_F)
    x = g[i].mean()
    print("Default Status",x)
    b = df[i].corr(df[TARGET_F])
    b = round(100*b,1)
    print("Default Correlation = ", b, "%")
    c = df[i].corr(df[TARGET_A])
    c = round(100*c,1)
    print("Loss Amount Correlation = ", c, "%")
    print("===============\n")


#Exploring statistics for float variables.
print("===============\n")
print("FLOAT VARIABLES" )
print("\n")
for i in floatList :
    print("Variable=",i)
    g = df.groupby(TARGET_F)
    x = g[i].mean()
    print("Default Status", x)  
    b = df[i].corr(df[TARGET_F])
    b = round(100*b,1)
    print("Default Correlation = ", b, "%")
    c = df[i].corr(df[TARGET_A])
    c = round(100*c, 1)
    print("Loss Amount Correlation = ", c, "%")
    print("===============\n")


#PLotting pie charts for object values.
for i in objList :
    x = df[i].value_counts(dropna=False)
    theLabels = x.axes[0].tolist()
    theSlices = list(x)
    plt.pie(theSlices,
            labels=theLabels,
             startangle = 90,
             shadow=False,
             autopct="%1.1f%%")
    plt.title("Pie Chart: " + i)
    plt.show()
    print("=====\n")


#PLotting histogram for int values.
for i in intList:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()


#PLotting histograms for float values.
for i in floatList :
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()



#Looking for relationships between the variables and the loss amount.
for i in range(0, len(df.columns),5):
    sns.pairplot(df, y_vars = ['TARGET_LOSS_AMT'], x_vars =df.columns[i:i+5])


#Because the binary object REASON has missing values and we are going to use one hot encoding, I will fill the missing values with the mode instead of with MISSING.
print("===============\n\n")
print("MISSING VALUES:\n\n")
for i in objList :
    if df[i].isna().sum() == 0 : continue
    print(i) 
    print("HAS MISSING")
    NAME = "IMP_"+i
    print(NAME) 
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna(df[NAME].mode()[0])
    print("variable",i," has this many missing", df[i].isna().sum())
    print("variable",NAME," has this many missing", df[NAME].isna().sum())
    g = df.groupby(NAME)
    print(g[NAME].count())
    print("\n")
    df = df.drop(i, axis=1)



#One hot encoding binary variables.
df["IMP_REASON_DebtCon_Yes"] = (df["IMP_REASON"].isin(["DebtCon"])+0)
df = df.drop("IMP_REASON", axis=1)


#One hot encoding the categorical variable IMP_JOB.
dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A ]) : continue
    if dt[i] in (["object"]) : objList.append(i)

for i in objList:
    thePrefix = "z_" + i
    y = pd.get_dummies(df[i], prefix=thePrefix, dummy_na=False)   
    df = pd.concat([df, y], axis=1)
    df = df.drop(i, axis=1)



#Imputing all missing nuerical data and creating a flag variable to indicate if the value was missing.
for i in numList :
    if df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    df[FLAG] = df[i].isna() + 0
    df[IMP] = df[ i ]
    df.loc[df[IMP].isna(), IMP] = df[i].median()
    df = df.drop(i, axis=1)


#Sorting the data.
df = df[sorted(df.columns)]

theColumn = df.pop(TARGET_F)  
df.insert(0, TARGET_F, theColumn) 

theColumn = df.pop( TARGET_A )  
df.insert(1, TARGET_A, theColumn) 

print("===============\n\n")
print("FIRST THREE ENTRIES:\n\n")
print(df.head(3))





