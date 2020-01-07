from sklearn.neighbors import KNeighborsClassifier
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

    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric='manhattan')
    neigh.fit(x_train, y_train)

    return (metrics.accuracy_score(y_test, neigh.predict(x_test)),
            metrics.accuracy_score(y_train, neigh.predict(x_train)))

def make_plot(ratios, accuracies, title):
    plt.figure()
    plt.plot(ratios, [acc[0] for acc in accuracies], label='test data')
    plt.plot(ratios, [acc[1] for acc in accuracies], label='train data')
    plt.xlabel('training data size')
    plt.ylabel('accuracy')
    plt.title(f'{title}\naccuracy(training data size)')
    plt.legend()
    plt.savefig(f'{title}.png')

def tic_tac_toe():
    features, targets = [], []
    with open("Tic_tac_toe.txt") as inp:
        for line in inp:
            features.append(line.split(',')[0:9])
            targets.append(line.split(',')[9].strip())

    le = preprocessing.LabelEncoder()
    features_encoded = [le.fit_transform(sample) for sample in features]
    targets_encoded = le.fit_transform(targets)

    ratios = np.linspace(0.1, 0.9, 20)
    accuracies = [calculate_accuracy(features_encoded, targets_encoded, ratio) for ratio in ratios]
    make_plot(ratios, accuracies, 'tic-tac-toe')

def spam():
    df = pd.read_csv('spam.csv', sep=',')
    features = df.iloc[:, 1:58].values
    targets = df['type'].values
    targets_encoded = preprocessing.LabelEncoder().fit_transform(targets)

    ratios = np.linspace(0.1, 0.9, 20)
    accuracies = [calculate_accuracy(features, targets_encoded, ratio) for ratio in ratios]
    make_plot(ratios, accuracies, 'spam')

def glass():
    df = pd.read_csv('glass.csv', sep=',')
    features = df.iloc[:, 2:-1].values
    targets = df['Type'].values

    plt.figure()
    metrics = ['euclidean', 'manhattan', 'chebyshev'] #, 'minkowski']
    max_neighbors = 30
    for metric in metrics:
        scores = []
        for n_neigbors in range (1, max_neighbors):
            neigh = KNeighborsClassifier(n_neighbors=n_neigbors, n_jobs=1, metric=metric)
            scores.append(cross_val_score(neigh, features, targets, cv=5, n_jobs=1).mean())
        plt.plot(range(1, max_neighbors, 1), scores, label=metric)
    plt.title('Glass')
    plt.xlabel('n_neighbors')
    plt.ylabel('accuracy')
    plt.legend(title='Distance Metric:');


    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=1, metric='manhattan')
    neigh.fit(features, targets)
    print('prediction for RI=1.516 Na=11.7 Mg=1.01 Al=1.19 Si=72.59 K=0.43 Ca=11.44 Ba=0.02 Fe=0.1:')
    sample = [1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]
    print(neigh.predict([sample]))
    print(neigh.predict_proba([sample]))


    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=1, metric='manhattan')
    accuracy = cross_val_score(neigh, features, targets, cv=5, n_jobs=1).mean()
    print('Change in accuracy by excluding column:')
    columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
    for col in columns:
        features = df.sub(df[col], axis=0).iloc[:, 2:-1].values
        neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=1, metric='manhattan')
        new_accuracy = cross_val_score(neigh, features, targets, cv=5, n_jobs=1).mean()
        print(col,': ', new_accuracy - accuracy)

def svmdata4():
    df_train = pd.read_csv('svmdata4.txt', delim_whitespace=True)
    df_test = pd.read_csv('svmdata4test.txt', delim_whitespace=True)

    dict = {'green': 1, 'red': 2}
    x_train = df_train[['X1', 'X2']].values
    x_test = df_test[['X1', 'X2']].values
    y_train = [dict[val] for val in df_train['Colors'].values]
    y_test = [dict[val] for val in df_test['Colors'].values]

    plt.figure()
    knn_metrics = ['euclidean', 'manhattan', 'chebyshev']
    max_neighbors = 40
    for metric in knn_metrics:
        scores = []
        for n_neigbors in range (1, max_neighbors):
            neigh = KNeighborsClassifier(n_neighbors=n_neigbors, n_jobs=1, metric=metric)
            neigh.fit(x_train, y_train)
            scores.append(metrics.accuracy_score(y_test, neigh.predict(x_test)))
        plt.plot(range(1, max_neighbors, 1), scores, label = metric)
    plt.title('svmdata4')
    plt.xlabel('n_neighbors')
    plt.ylabel('accuracy')
    plt.legend(title='Distance Metric:')


    neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=1, metric='manhattan')
    neigh.fit(x_train, y_train)
    x_merged, y_merged = np.array([*x_train, *x_test]), np.array([*y_train, *y_test])
    pred_correct = np.isclose(neigh.predict(x_merged), y_merged)

    inv_dict = {v: k for k, v in dict.items()}
    x1, x2 = np.array([v[0] for v in x_merged]), np.array([v[1] for v in x_merged])
    plt.figure()
    plt.scatter(x1[pred_correct==True], x2[pred_correct==True],
                color=[inv_dict[v] for v in y_merged[pred_correct==True]], marker='o')
    plt.scatter(x1[pred_correct==False], x2[pred_correct==False],
                color=[inv_dict[v] for v in y_merged[pred_correct==False]], marker='x')
    plt.xlabel('x1')
    plt.ylabel('x2')

tic_tac_toe()
spam()
glass()
svmdata4()
plt.show()