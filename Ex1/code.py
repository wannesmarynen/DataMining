import math

import pandas as pandas
from sklearn.compose import make_column_transformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def findBestClassifier():
    dataframe = pandas.read_excel('Ex1/data/existing-customers.xlsx')

    ohe = OneHotEncoder(sparse_output=False)

    y = dataframe.iloc[:, -1]
    X = dataframe.iloc[:, :-1]
    # print(X)

    column_transform = make_column_transformer(
        (ohe,
         ['workclass', 'marital-status', 'occupation', 'relationship', 'sex', 'native-country'])
    )

    dataframe.drop(columns=['education', 'RowID'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=55)

    classifiers = [
        ('KNeighborsClassifier', KNeighborsClassifier(3)),
        ('SVC', SVC(kernel="linear", C=0.025)),
        ('SVC', SVC(gamma=2, C=1)),
        ('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5)),
        ('MLPClassifier', MLPClassifier(alpha=1, max_iter=1000)),
        ('AdaBoostClassifier', AdaBoostClassifier()),
        ('GaussianNB', GaussianNB()),
        ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()),
    ]

    bestClassifier = None
    falsePositiveRate = 100
    revenue = math.inf * -1

    for name, classifier in classifiers:
        lm_pipeline = make_pipeline(column_transform, classifier)

        lm_pipeline.fit(X_train, Y_train)
        predictions = lm_pipeline.predict(X_test)
        matrix = confusion_matrix(Y_test, predictions)
        print(name)
        truepositivecost = matrix[1, 1] * 0.1 * 980
        falsepositivecost = matrix[0, 1] * 0.05 * -310
        packagecost = (matrix[1, 1] + matrix[0, 1]) * -10
        print('truepositivecost= ' + truepositivecost.__str__())
        print('falsepositivecost= ' + falsepositivecost.__str__())
        print('packagecost= ' + packagecost.__str__())
        print('cost = ' + (truepositivecost + falsepositivecost + packagecost).__round__(2).__str__())
        cost = (truepositivecost + falsepositivecost + packagecost).__round__(2)
        if cost > revenue:
            print(f'BEST: {name}')
            revenue = cost
            falsePositiveRate = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
            bestClassifier = lm_pipeline
        print()
    return falsePositiveRate, bestClassifier


def calculateExpectedReward(pipeline, falsepositiveRate):
    dataframe2 = pandas.read_excel('Ex1/data/potential-customers.xlsx')
    rowIDS = dataframe2['RowID']

    dataframe2.drop(columns=['education', 'RowID'])

    predictions = pipeline.predict(dataframe2)
    positivePredictions = 0
    with open('data/IDS.txt', 'w') as f:
        for index, prediction in enumerate(predictions):
            if prediction == '>50K':
                # print(rowIDS[index])
                f.write(rowIDS[index] + '\n')
                positivePredictions+=1
    expexctedTruePositive = positivePredictions * (1 - falsepositiveRate)
    expexctedFalseNegative = positivePredictions * falsepositiveRate
    truepositivecost = expexctedTruePositive * 0.1 * 980
    falsepositivecost = expexctedFalseNegative * 0.05 * -310
    packagecost = positivePredictions * -10
    print('truepositivecost= ' + truepositivecost.__str__())
    print('falsepositivecost= ' + falsepositivecost.__str__())
    print('packagecost= ' + packagecost.__str__())
    print('cost = ' + (truepositivecost + falsepositivecost + packagecost).__round__(2).__str__() + '\n')


if __name__ == '__main__':
    falsepositverate, classifier = findBestClassifier()
    calculateExpectedReward(classifier, falsepositverate)
