import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc3 import forestplot
import seaborn as sns


# # Importing data
df = pd.read_csv(r'-------------')
print(df)
X = df.iloc[0:9, 1:86]
print(X)
Y = df.iloc[9, 1:86]
print(Y)
if __name__ == '__main__':
    with pm.Model() as model_mlr:
        alpha_tmp = pm.Normal('alpha_tmp', mu=6, sd=8)
        beta = pm.Normal('beta', mu=0, sd=5, shape=9)

        epsilon = pm.Exponential('epsilon', 5)

        nu = pm.Exponential('nu', 0.1)
        mu = alpha_tmp + pm.math.dot(beta, X)

        alpha = pm.Deterministic('mu', alpha_tmp + pm.math.dot(beta, X))

        y1 = pm.StudentT('y_pred', mu=mu, sd=epsilon, nu=nu, observed=Y)


        data_mu = y1.tag.test_value
        data_mu = pd.DataFrame(data_mu)
        print(y1.tag.test_value)
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace_mlr = pm.sample(draws=2000, step=step, start=start, random_seed=123, chain=4, tune=1000)

varsname = ['beta']
pm.traceplot(trace_mlr, varsname)
plt.show()

forestplot(trace_mlr, varnames=['beta'], alpha=0.05)
plt.show()

data = pm.df_summary(trace_mlr, varsname)
# 保存摘要记录
data.to_csv(r'---------')
print(data)

# Post-testing

# Post-assay sampling
pcc_tlr = pm.sample_ppc(trace_mlr, samples=1000, model=model_mlr, random_seed=123)
# print(pcc_tlr)
# Model predictions
ys = pcc_tlr['y_pred']
y_hpd = pm.stats.hpd(ys, alpha=0.2)
# print(y_hpd)
ys = pd.DataFrame(ys)
print(ys)
# # Save data
# ys.to_csv(r'---------')
# Mapping of model results
avys = np.average(ys, axis=0)
avys = pd.DataFrame(avys)
sns.distplot(avys, color='b', axlabel='10s')
plt.show()
plt.scatter(Y, avys)
plt.show()
