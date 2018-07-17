import csv
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def createExamples():
    header=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,'clase']
    data=[]
    numbersWeHave = range(0,10)
    for eachNum in numbersWeHave:
        for furtherNum in range(1,10):
            imgFilePath = 'images/numbers/'+str(eachNum)+'.'+str(furtherNum)+'.png'
            ei = Image.open(imgFilePath)
            ei = np.matrix(ei.convert('L')).flatten().tolist()
            ei=ei[0]
            ei.append(eachNum)
            data.append(ei)
    numberArrayExamples = open('numArEx.csv','w')
    with numberArrayExamples:
        writer = csv.writer(numberArrayExamples)
        writer.writerow(header)
        writer.writerows(data)

def createModel():
    # print("Here's the info from the DataSet:")
    dataframe = pd.read_csv("numArEx.csv")
    # print(dataframe.head())
    # print(dataframe.describe())
    # print(dataframe.groupby(['clase']).size())

    X = np.array(dataframe.drop(['clase'],1))
    y = np.array(dataframe['clase'])

    # print(X.shape)

    model = linear_model.LogisticRegression()
    model.fit(X,y)

    predictions = model.predict(X)
    print(predictions[0:5])
    print(model.score(X,y))
    return X, y, model

def createModelWcv():
    X, y, model = createModel()

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size = validation_size, random_state = seed)
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    print(cv_results.mean(), cv_results.std())

    predictions = model.predict(X_validation)

    return Y_validation, predictions, model

def results():
    Y_validation, predictions, model = createModelWcv()
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    return model

def predict(imgFilePath):
    model = results()
    dict = {}
    ei = Image.open(imgFilePath)
    ei = np.matrix(ei.convert('L')).flatten().tolist()
    ei=ei[0]
    for x in range(0,len(ei)):
        dict[str(x+1)]=[ei[x]]
    print(dict)
    X_new = pd.DataFrame(dict)
    print("Ese numero predigo que es: " + str(model.predict(X_new)))

imgFilePath = 'images/test2.png'
predict(imgFilePath)
