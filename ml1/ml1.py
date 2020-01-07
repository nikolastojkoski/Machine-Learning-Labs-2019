from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def calculate_accuracy(features, targets, train_size):
    test_size = 1 - train_size
    x_train, x_test, y_train, y_test = \
        train_test_split(features, targets, test_size=test_size, random_state=1)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    #all_x, all_y = [*x_train, *x_test], [*y_train, *y_test]

    return (metrics.accuracy_score(y_test, gnb.predict(x_test)),
            metrics.accuracy_score(y_train, gnb.predict(x_train)))
            #metrics.accuracy_score(all_y, gnb.predict(all_x)))

def make_plot(ratios, accuracies, title):
    plt.figure()
    plt.plot(ratios, [acc[0] for acc in accuracies], label='test data')
    plt.plot(ratios, [acc[1] for acc in accuracies], label='train data')
    #plt.plot(ratios, [acc[2] for acc in accuracies], label='all data')
    plt.xlabel('training data size')
    plt.ylabel('accuracy')
    plt.title(f'{title}\naccuracy(training data size)')
    plt.legend()
    #plt.savefig(f'{title}.png')

def tic_tac_toe():
    features, targets = [], []
    with open("Tic_tac_toe.txt") as inp:
        for line in inp:
            features.append(line.split(',')[0:9])
            targets.append(line.split(',')[9].strip())

    le = preprocessing.LabelEncoder()
    features_encoded = [le.fit_transform(sample) for sample in features]
    targets_encoded = le.fit_transform(targets)

    ratios = np.linspace(0.01, 0.9, 100)
    accuracies = [calculate_accuracy(features_encoded, targets_encoded, ratio) for ratio in ratios]
    make_plot(ratios, accuracies, 'tic-tac-toe')

def spam():
    df = pd.read_csv('spam.csv', sep=',')
    features = df.iloc[:, 1:58].values
    targets = df['type'].values
    targets_encoded = preprocessing.LabelEncoder().fit_transform(targets)

    ratios = np.linspace(0.001, 0.9, 100)
    accuracies = [calculate_accuracy(features, targets_encoded, ratio) for ratio in ratios]
    make_plot(ratios, accuracies, 'spam')

def x1x2():
    np.random.seed(25)
    x1 = [*np.random.normal(10, 4, 50), *np.random.normal(20, 3, 50)]
    x2 = [*np.random.normal(14, 4, 50), *np.random.normal(18, 3, 50)]

    colors = ['red']*50 + ['blue']*50

    plt.figure()
    plt.scatter(x1, x2, color=['red']*50 + ['blue']*50)
    plt.xlabel('x1')
    plt.ylabel('x2')

    x = np.vstack((x1, x2)).T
    y = np.array([-1]*50 + [1]*50)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    colors = np.array([colors[indices[i]] for i in range(x.shape[0])])
    scores = cross_val_score(GaussianNB(), x, y, cv = 5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = GaussianNB()
    clf.fit(x, y)
    x1, x2 = np.array([v[0] for v in x]), np.array([v[1] for v in x])
    pred_correct = np.isclose(y, clf.predict(x))

    plt.figure()
    plt.scatter(x1[pred_correct==True], x2[pred_correct==True],
                color=colors[pred_correct==True], marker='o')
    plt.scatter(x1[pred_correct==False], x2[pred_correct==False],
                color=colors[pred_correct==False], marker='x')
    plt.xlabel('x1')
    plt.ylabel('x2')


tic_tac_toe()
spam()
x1x2()
plt.show()