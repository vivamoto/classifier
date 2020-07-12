# -*- coding: utf-8 -*-
"""
Created on Fri May 22, 2020

@author: Victor Ivamoto


Disciplina de Aprendizado de Máquina

Atividade No 1 - Classificador Linear

Implementar uma rede neural com função de ativação na saída softmax
para iris, diabetes e hepatite.

Comparar o resultado obtido com os modelos lineares, svm e tsvm.

Data da entrega: 26/05/2020

====================================
Origem dos datasets:

hepatitis:
https://archive.ics.uci.edu/ml/datasets/Hepatitis

Iris:
https://archive.ics.uci.edu/ml/datasets/Iris

Diabetes
https://www.kaggle.com/uciml/pima-indians-diabetes-database
"""

# Altera a pasta de trabalho
cd "D:\\Documentos\\Profissão e Carreira\\Mestrado\\Aprendizado de Máquina\\Exercícios\\Neural Networks"
pdir = "D:\\Documentos\\Profissão e Carreira\\Mestrado\\Aprendizado de Máquina\\Exercícios\\Neural Networks\\"

import numpy as np
import pandas as pd
import requests
import os.path

#======================
# 1.1 Download datasets
#======================

# This function downloads the datasets from UCI site
# Since the diabetes dataset is in Kaggle, we are unable
# to download automatically. Access the site, download the file
# and save in this code working directory.
def getDataset(url, folder, fname):

  # Create folder to save the files
  if not os.path.exists(folder):
    os.mkdir(folder)

  # Download file and store in variable "myfile"
  myfile = requests.get(url)

  # Save file in disk
  open(folder + "/" + fname, 'wb').write(myfile.content)

# Download hepatitis dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
getDataset(url = url, folder = "hepatitis", fname = "hepatitis.data")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.names"
getDataset(url = url, folder = "hepatitis", fname = "hepatitis.names")

# Download iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
getDataset(url = url, folder = "iris", fname = "iris.data")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names"
getDataset(url = url, folder = "iris", fname = "iris.names")

# Load the datasets
iris     = pd.read_csv("iris/iris.data", names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
hepatitis = pd.read_csv("hepatitis/hepatitis.data", names = ["Class", "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY", ])

diab_train = pd.read_csv("diabetes/dataset_train.txt", sep = '\t')
diab_test  = pd.read_csv("diabetes/dataset_test.txt",  sep = '\t')

results = pd.DataFrame()

# Inspect the imported datasets
diab_train.head()
diab_test.head()
hepatitis.head()
iris.head()

#======================
# 1.2 Outcome: Multi-class classification - 1 of n
#======================
# We create a column for each class and use the 1 of n approach
#       X.w = Y
# Where:
#   X is the matrix of features,
#   w is the weight vector and
#   Y is the outcome vector.
#
# In this activity, only the iris dataset is multi class.
# Iris: create a column for each flower class, using 1 of n
# For example, the 'setosa' column contains 1 for setosa observations and 0 otherwise
iris.insert(loc = iris.shape[1], column = "setosa", value = (iris["class"] == "Iris-setosa").astype('int') , allow_duplicates = True)
iris.insert(loc = iris.shape[1], column = "versicolor", value = (iris["class"] == "Iris-versicolor").astype('int') , allow_duplicates = True)
iris.insert(loc = iris.shape[1], column = "virginica", value = (iris["class"] == "Iris-virginica").astype('int') , allow_duplicates = True)

iris.head()
iris.dtypes

# Convert columns to real numbers
# Return the columns not converted
def real_num(ds):
    col = []
    for i in range(1, ds.shape[1]):
        try:
            ds.iloc[:, i] = ds.iloc[:, i].astype(float)
        except:
            col.append(i)
            print('Column ' + str(i) + ' contains "?"')
    return ds, col

#======================
# 1.3 Missing values
#======================
# The hepatitis dataset contains many missing values, represented by "?"
# Since these "?" are spread in the dataset, we'll remove them carefully.

# Before deleting any row or column, let's check the matrix size, so
# we can compare later.
hepatitis.shape

# The first approach is to remove the rows and columns with the maximum
# number of "?".

# Drop rows with more than 20% missing values
# We store the row numbers in the "row" variable, and later we drop
# these rows.
i = 0       # counter to loop over all rows
row = []    # Stores the row numbers to delete
while i < hepatitis.shape[0]:
    # Find row with more than 20% of "?"
    if np.mean(hepatitis.iloc[i,:] == "?") > 0.2:
        row.append(i)
    i = i + 1
# Delete the rows
hepatitis.drop(row, inplace = True)

# Drop columns with more than 20% missing values
# We do the same as above, but for columns.
# We save the column numbers in "col" variable, and later
# we delete these columns.
i = 0
col = []    # stores column numbers
while i < hepatitis.shape[1]:
    # Find columns to delete
    if np.mean(hepatitis.iloc[:,i] == "?") > 0.2:
        col.append(i)
    i = i + 1
# Delete columns
hepatitis.drop(columns = hepatitis.columns[col], inplace = True)

# The numeric columns, we replace the missing values with the mean
# value of the feature.
# Loop over all dataframe columns
for i in range(14, 18):

    # Calculate the column mean
    new_value = hepatitis[hepatitis.iloc[:,i] != "?"].iloc[:,i].astype(float).mean()

    # Replace the "?" value with the new value
    hepatitis.iloc[:,i].replace(to_replace = "?", value = new_value, inplace = True)

    # Convert the column to float
    hepatitis.iloc[:,i] = hepatitis.iloc[:,i].astype(float)


hepatitis, col = real_num(hepatitis)
# Count the number of "?" remaining in each column
np.sum(hepatitis.iloc[:,col] == "?")
col
hepatitis.loc[hepatitis.iloc[:,3] == "?", :].iloc[:,3:10]
hepatitis.loc[hepatitis.iloc[:,8] == "?", :].iloc[:,3:10]
hepatitis.loc[hepatitis.iloc[:,9] == "?", :].iloc[:,3:10]
hepatitis.dtypes

# There are 6 rows with missing values
row = hepatitis.loc[hepatitis.iloc[:,3] == "?", :].index
hepatitis.drop(row, inplace = True)

row = hepatitis.loc[hepatitis.iloc[:,9] == "?", :].index
hepatitis.drop(row, inplace = True)

# Convert columns to real numbers
hepatitis, col = real_num(hepatitis)

print(hepatitis.shape)

# Inspect data types
print(hepatitis.dtypes)

#======================
# 1.4 Categorical variables: -1 and 1
#======================
# Regression uses the signal function to calculate the outcome
# So, we convert all categorical values to 0 and 1

# Hepatitis: convert categorical variables with 1 and 2 to 0 and 1.
for i in ['Class', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']:
    hepatitis[i] = hepatitis[i] - 1

iris.to_csv(pdir + 'iris\\iris.csv', index = False, decimal = '.', sep = '\t')
hepatitis.to_csv(pdir + 'hepatitis\\hepatitis.csv', index = False, decimal = '.', sep = '\t')

