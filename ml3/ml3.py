import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz-2.38/bin/'
import graphviz

def train_test_check(x, y, criterion, parameter, parameter_values):
    accuracies = []
    for pval in parameter_values:
        d = {parameter: pval}
        clf = tree.DecisionTreeClassifier(criterion=criterion, **d, random_state=1)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        scores = cross_validate(clf, x, y, scoring='accuracy', return_train_score=True, cv=kfold)
        accuracies.append((scores['test_score'].mean(), scores['train_score'].mean()))
    plt.figure()
    plt.plot(parameter_values, [acc[0] for acc in accuracies], label='test data')
    plt.plot(parameter_values, [acc[1] for acc in accuracies], label='train data')
    plt.xlabel(parameter)
    plt.ylabel('accuracy')
    plt.title(f'glass.csv; criterion = {criterion}')
    plt.legend()

def check2d(x, y, parameter, parameter_values):
    plt.figure()
    for criterion in ['gini', 'entropy']:
        accuracies = []
        for pval in parameter_values:
            d = {parameter: pval}
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            clf = tree.DecisionTreeClassifier(criterion=criterion, **d, random_state=1)
            accuracies.append(cross_val_score(clf, x, y, cv=kfold, scoring='accuracy').mean())
        plt.plot(parameter_values, accuracies, label=criterion)
    plt.xlabel(parameter)
    plt.ylabel('accuracy')
    plt.title('Glass')
    plt.legend(title='criterion:')


def check3d(x, y, param1, param2, p1_values, p2_values, criterion='entropy', cmap=cm.autumn):
    zvals, xvals, yvals = [], [], []
    for p1v in p1_values:
        for p2v in p2_values:
            d = {param1: p1v, param2: p2v}
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            clf = tree.DecisionTreeClassifier(criterion=criterion, **d, random_state=1)
            zvals.append(cross_val_score(clf, x, y, cv=kfold).mean())
            xvals.append(p1v)
            yvals.append(p2v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(xvals, yvals, zvals, cmap=cmap, vmin=min(zvals), vmax=max(zvals))
    ax.set_zlabel('accuracy')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f'glass.csv; criterion={criterion}')

def glass():
    df = pd.read_csv('glass.csv')
    x = df.iloc[:, 2:-1].values
    y = df['Type'].values

    train_test_check(x, y, 'gini', 'max_leaf_nodes', range(5, 50))
    train_test_check(x, y, 'entropy', 'max_leaf_nodes', range(5, 50))

    check3d(x,y, 'max_features', 'max_leaf_nodes', range(1, 10, 1), range(5, 50, 5))
    check3d(x,y, 'max_depth', 'max_leaf_nodes', range(2, 15), range(5, 50, 5))
    check3d(x,y, 'max_features', 'max_leaf_nodes', range(1, 10, 1), range(5, 50, 5),criterion='gini', cmap=cm.winter)
    check3d(x,y, 'max_depth', 'max_leaf_nodes', range(2, 15), range(5, 50, 5), criterion='gini', cmap=cm.winter)

    train_test_check(x, y, 'gini', 'max_depth', range(2, 15))
    train_test_check(x, y, 'gini', 'max_features', range(2, 10))
    train_test_check(x, y, 'gini', 'min_samples_split', np.linspace(0.01, 1, 20))
    train_test_check(x, y, 'gini', 'min_samples_leaf', np.linspace(0.01, 0.5, 20))
    train_test_check(x, y, 'entropy', 'max_depth', range(2, 15))
    train_test_check(x, y, 'entropy', 'max_features', range(2, 10))
    train_test_check(x, y, 'entropy', 'min_samples_split', np.linspace(0.01, 1, 20))
    train_test_check(x, y, 'entropy', 'min_samples_leaf', np.linspace(0.01, 0.5, 20))

    check2d(x, y, 'max_depth', range(2, 15, 1))
    check2d(x, y, 'max_features', range(1, 10, 1))
    check3d(x, y, 'max_depth', 'min_samples_split', range(2, 15, 1), np.linspace(0.01, 1, 20))
    check3d(x, y, 'max_depth', 'min_samples_leaf', range(2, 15, 1), np.linspace(0.01, 0.5, 20))
    check3d(x, y, 'min_samples_split', 'min_samples_leaf', np.linspace(0.01, 1, 20), np.linspace(0.01, 0.5, 20))
    check3d(x, y, 'min_samples_split', 'min_samples_leaf', np.linspace(0.01, 1, 20), np.linspace(0.01, 0.5, 20),
            criterion='gini', cmap=cm.winter)
    check3d(x, y, 'max_depth', 'max_features', range(2, 15, 1), range(1, 10, 1),
            criterion='gini', cmap=cm.winter)

    #optimal
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_features=7, max_depth=6, random_state=1)
    print('accuracy: ', cross_val_score(clf, x, y, cv=kfold).mean())
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
      feature_names=df.columns.values[2:-1], class_names=['1', '2', '3', '5', '6', '7'],
      filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("glass_opt")

    #smallest
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=15, random_state=1)
    print('accuracy: ', cross_val_score(clf, x, y, cv=kfold).mean())
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
      feature_names=df.columns.values[2:-1], class_names=['1', '2', '3', '5', '6', '7'],
      filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("glass_md8_ml15")

    #default
    clf = tree.DecisionTreeClassifier(random_state=1)
    print('accuracy: ', cross_val_score(clf, x, y, cv=kfold).mean())
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
      feature_names=df.columns.values[2:-1], class_names=['1', '2', '3', '5', '6', '7'],
      filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("glass")

def lenses():
    df = pd.read_csv('Lenses.txt', delim_whitespace=True, header=None)

    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    clf = tree.DecisionTreeClassifier()
    clf.fit(x, y)
    print(clf.predict([[2, 1, 2, 1]]))

    dot_data = tree.export_graphviz(clf, out_file=None,
        feature_names=['возраст', 'состояние зрения', 'астигматизм', 'состояние слезы'],
        class_names=['жесткие линзы', 'мягкие линзы', 'не следует носить линзы'],
        filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("Lenses")

    from sklearn.preprocessing import OneHotEncoder
    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(x)
    clf = tree.DecisionTreeClassifier()
    clf.fit(enc.transform(x).toarray(), y)

    print(clf.predict(enc.transform([[2,1,2,1]]).toarray()))

    feature_names = ['молодой', 'предстарческая дальнозоркость', 'старческая дальнозоркость',
                     'близорукий', 'дальнозоркий',
                     'астигматизм-да', 'астигматизм-нет',
                     'слезы-сокращенная', 'слезы-нормаьная']
    dot_data = tree.export_graphviz(clf, out_file=None,
        feature_names=feature_names,
        class_names=['жесткие линзы', 'мягкие линзы', 'не следует носить линзы'],
        filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("Lenses_onehot")

def spam7():
    df = pd.read_csv('spam7.csv')
    x = df.iloc[:, 0:-1].values
    y = df['yesno'].values

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 15),
        'max_features': range(1, 7)
    }
    clf = tree.DecisionTreeClassifier(random_state=1)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf_cv = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold)
    clf_cv.fit(x, y)
    print(clf_cv.best_params_)
    print(clf_cv.best_score_)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = tree.DecisionTreeClassifier(random_state=1)
    print('accuracy: ', cross_val_score(clf, x, y, cv=kfold).mean())

glass()
lenses()
spam7()
plt.show()





