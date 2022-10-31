import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.inspection import partial_dependence
from scipy.interpolate import splev, splrep



# Loading data
data = pd.read_csv(r'---------')
# Classification of independent and dependent variables

x_columns = []
for x in data.columns:
    if x not in ['LDI']:
        x_columns.append(x)
X = data[x_columns]
print(X)

y = data['LDI']
print(y)
# Dividing the data set
kf = KFold(n_splits=30, shuffle=False)
for train_index, valid_index in kf.split(X, y) :
    X_train, X_test = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_test = y[train_index], y[valid_index]

# Constructing classes for the GBDT algorithm

Regressor = ensemble.GradientBoostingRegressor(n_estimators=2000, max_depth=7, min_samples_split=4,
                                                 learning_rate=0.1, subsample=1, random_state=123)

# Fitting of the algorithm to the training data set

Regressor.fit(X_train, y_train)
# Predictions of the algorithm on the test dataset

pred1 = Regressor.predict(X_test)
print(pred1)
# Return the predicted effect of the model
# Model assessment

acc_train = Regressor.score(X_train, y_train)

acc_test = Regressor.score(X_test, y_test)
print(acc_train)
#
print(acc_test)
# Value of the degree of importance of the variable

importance = Regressor.feature_importances_
# Constructing sequences for mapping

Impt_Series = pd.Series(importance, index=X_train.columns)

dataplot = Impt_Series.sort_values(ascending=True)
dataplot.plot(kind='barh', color='k', alpha=0.7)
plt.show()
plt.savefig(r'-----------')
features = x_columns
print(features)
plot_df = pd.DataFrame(columns=['x', 'y'])
for i in features:
    sns.set_theme(style="ticks", palette="deep", font_scale = 1.3)
    fig = plt.figure(figsize=(6, 5), dpi=100)
    ax = plt.subplot(111)
    pdp = partial_dependence(Regressor, X_train, [i],
                            kind="average",
                            method = 'brute',
                            grid_resolution=200)
    plot_x = pd.Series(pdp['values'][0])
    plot_y = pd.Series(pdp['average'][0])
    tck = splrep(plot_x, plot_y, k=3, s=30) # ：https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html?highlight=interpolate
    xnew = np.linspace(plot_x.min(), plot_x.max(), 300)
    ynew = splev(xnew, tck, der=0)
    df_i = pd.concat([plot_x.rename('x'), plot_y.rename('y')], axis=1)
    plot_df = plot_df.append(df_i)
    print(plot_df)
    plt.scatter(plot_x, plot_y, color='k', alpha=0.4)
    plt.plot(xnew, ynew, linewidth=2)
    sns.rugplot(data=X_train.sample(40, replace=False), x=i, height=.06, color='k', alpha=0.3)

    x_min = plot_x.min()-(plot_x.max()-plot_x.min())*0.1
    x_max = plot_x.max()+(plot_x.max()-plot_x.min())*0.1
    plt.title('Partial Dependence Plot of '+i, fontsize=18)
    plt.ylabel('Partial Dependence')
    plt.xlim(x_min, x_max)
    plt.show()





