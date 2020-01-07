import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def task1():
    df = pd.read_csv('cygage.txt', delim_whitespace=True)
    X = np.array(df['Depth'].values).reshape(-1, 1)
    Y = df['calAge'].values
    w = df['Weight']

    reg = LinearRegression().fit(X, Y, w)
    print('LineaerRegression:')
    print(f' score = {reg.score(X, Y, w)}')
    print(f' coef_ = {reg.coef_}')
    print(f' intercept = {reg.intercept_}')

    plt.figure()
    plt.plot(range(1, 1000), reg.predict(np.array(range(1, 1000)).reshape(-1, 1)), c='blue')
    plt.scatter(X, Y, c='red')


def findBestFeatures(X, Y, print_all=False):

    bestFeatures, bestRSS = None, np.inf

    for k in range(1, X.shape[1] + 1):

        idx_combinations = list(itertools.combinations(range(0, X.shape[1]), k))
        for comb in idx_combinations:

            reg = LinearRegression().fit(X[:, list(comb)], Y)
            yy = reg.predict(X[:, list(comb)])
            RSS = np.sum([(Y[i] - yy[i])**2 for i in range(len(Y))])

            if RSS < bestRSS:
                bestRSS, bestFeatures = RSS, list(comb)
            if print_all:
                print('idx_comb = {}, RSS = {:.2f}'.format(comb, RSS))

    return bestFeatures

def task2():
    df = pd.read_csv('reglab.txt', delim_whitespace=True)
    X = np.array(df.iloc[:, 1:].values)
    Y = df['y'].values

    best_features = findBestFeatures(X, Y, print_all=True)
    print(f'best features: {best_features}')
    X = X[:, best_features]

    reg = LinearRegression().fit(X, Y)
    print(f'score = {reg.score(X, Y)}')

def task3():
    df = pd.read_csv('longley.csv')
    df = df.drop('Population', 1)

    Y = df['Unemployed'].values
    X = df.drop('Unemployed', 1).values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1)

    test_errors, train_errors = [], []
    for i in range(26):
        alpha = 10**(-3+0.2*i)
        reg = Ridge(alpha=alpha).fit(x_train, y_train)
        test_errors.append(1 - reg.score(x_test, y_test))
        train_errors.append(1 - reg.score(x_train, y_train))

    print([10**(-3+0.2*i) for i in range(26)])
    plt.figure()
    plt.plot([10**(-3+0.2*i) for i in range(26)], test_errors, label='test error')
    plt.plot([10**(-3+0.2*i) for i in range(26)], train_errors, label='train error')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.legend()
    plt.title('longley.csv;')

def task4():
    df = pd.read_csv('eustock.csv')
    x = np.array(range(df.shape[0])).reshape(-1, 1)

    plt.figure()
    for i in range(4):
        y = df.iloc[:, i].values
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)
        rss = np.sum([(y[i] - y_pred[i])**2 for i in range(len(y))])
        print(f'data = {df.columns[i]}', 'rss = {:.3e}'.format(rss))
        plt.plot(x, reg.predict(x), color=f'C{i}')
        plt.plot(x, df.iloc[:, i], color=f'C{i}', label=df.columns[i])
    plt.legend()

    y = [np.mean(df.iloc[i, :]) for i in range(df.shape[0])]
    reg = LinearRegression().fit(x, y)

    plt.figure()
    for i in range(4):
        plt.plot(x, df.iloc[:, i], label=f'{df.columns[i]}')
    plt.plot(x, reg.predict(x), color='b')
    plt.legend()
    plt.title(f'eustock.csv - mean')

def task5():
    df = pd.read_csv('JohnsonJohnson.csv')
    x = np.array(range(1960, 1981)).reshape(-1, 1)
    q = [[df.iloc[i, 1] for i in range(j, df.shape[0], 4)] for j in range(4)]

    plt.figure()
    for i in range(4):
        reg = LinearRegression().fit(x, q[i])

        y_pred = reg.predict(x)
        rss = np.sum([(q[i][j] - y_pred[j])**2 for j in range(len(q[i]))])
        print(f'Q{i+1}', 'rss = {:.3e}'.format(rss))
        print(f'Q{i+1}, prediction for 2016: {reg.predict([[2016]])}')
        plt.plot(x, q[i], color=f'C{i}')
        plt.plot(x, reg.predict(x), color=f'C{i}', label=f'Q{i+1}')
    plt.legend()

    qmean = [np.mean([q[j][i] for j in range(4)]) for i in range(x.shape[0])]
    plt.figure()
    for i in range(4):
        plt.plot(x, q[i], label=f'Q{i+1}')
    reg = LinearRegression().fit(x, qmean)
    plt.plot(x, reg.predict(x), color='blue')
    plt.legend()
    print(f'Mean, prediction for 2016: {reg.predict([[2016]])}')


def task6():
    df = pd.read_csv('sunspot.year.csv')
    x, y = np.array(df['index'].values).reshape(-1, 1), df['value'].values
    reg = LinearRegression().fit(x, y)
    plt.figure()
    plt.scatter(x, y, marker='.')
    plt.plot(x, reg.predict(x))
    plt.xlabel('year')
    plt.ylabel('sunspots')

def task7():
    df = pd.read_csv('cars.csv')
    x, y = np.array(df['speed'].values).reshape(-1, 1), df['dist'].values
    reg = LinearRegression().fit(x, y)
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, reg.predict(x))
    plt.xlabel('speed')
    plt.ylabel('break distance')
    print(f'speed = {40}, break distance = {reg.predict([[40]])}')

def task8():
    df = pd.read_csv('svmdata6.txt', delim_whitespace=True)
    x = np.array(df['X'].values).reshape(-1, 1)
    y = df['Y'].values

    eps_vals, rss_vals = np.linspace(0, 1, 5), []
    for eps in eps_vals:
        reg = SVR(C=1, kernel='rbf', epsilon=eps, gamma='auto').fit(x, y)
        y_pred = reg.predict(x)
        rss_vals.append(np.sum([(y[i] - y_pred[i])**2 for i in range(len(y))]))

        plt.figure()
        plt.scatter(x, y)
        plt.plot(x, reg.predict(x))
        plt.title(f'svmdata6.txt; SVR:epsilon={eps}')

    plt.figure()
    plt.plot(eps_vals, rss_vals)
    plt.xlabel('epsilon')
    plt.ylabel('RSS')

def task9():
    df = pd.read_csv('nsw74psid1.csv')
    x = df.iloc[:, 0:-1].values
    y = df['re78'].values

    regs = {'DecisionTree': DecisionTreeRegressor(),
            'SVR': SVR(gamma='auto'),
            'LinearRegression': LinearRegression()}
    for (key, reg) in regs.items():
        reg.fit(x, y)
        y_pred = reg.predict(x)
        rss = np.sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))])
        print(key, 'score = {:.3f}, RSS ={:.2e}'.format(reg.score(x, y), rss))

#task1()
#task2()
#task3()
#task4()
#task5()
#task6()
#task7()
#task8()
task9()


plt.show()