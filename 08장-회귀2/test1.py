import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

print(df.head(3))

y_target = df['PRICE']
x_data = df.drop(['PRICE'], axis=1, inplace=False)

print("Ridge \n")

ridge = Ridge(alpha=10)
neg_mse_scores = cross_val_score(ridge, x_data, y_target, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print("neg_mse_scores : ", neg_mse_scores)
print("rmse_scores : ", rmse_scores)
print("avg_rmse : ", avg_rmse)

print("\n")

alphas = [0, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    neg_mse_scores = cross_val_score(ridge, x_data, y_target, scoring="neg_mean_squared_error", cv=5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print("alpha : %d, avg_rmse : %f" % (alpha, avg_rmse))

fig, axs = plt.subplots(figsize=(18, 6), nrows=1, ncols=5)
coeff_df = pd.DataFrame()

for pos, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_data, y_target)
    coeff = pd.Series(data=ridge.coef_, index=x_data.columns)
    colname = 'alpha : ' + str(alpha)
    coeff_df[colname] = coeff
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3, 6)
    sns.barplot(x=coeff.values, y=coeff.index, ax=axs[pos])
plt.show()

print("\n")

sort_column = 'alpha : ' + str(alphas[0])
coeff_df.sort_values(by=sort_column, ascending=False)
print("coeff_df \n", coeff_df)

print("\n")

print("Lasso \n")

def get_linear_reg_eval(model_name, params=None, x_data_n=None, index_n=None, y_target_n=None):
    coeff_df = pd.DataFrame()
    for param in params:
        if model_name == 'Ridge' : model = Ridge(alpha=param)
        elif model_name == 'Lasso' : model = Lasso(alpha=param)
        elif model_name == 'ElasticNet' : model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, x_data_n, y_target_n, scoring="neg_mean_squared_error", cv=5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print("param : %f, avg_rmse : %f" % (param, avg_rmse))
        model.fit(x_data_n, y_target_n)
        coeff = pd.Series(data=model.coef_, index=index_n)
        colname = 'alpha : ' + str(param)
        coeff_df[colname] = coeff
    return coeff_df

lasso_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df = get_linear_reg_eval('Lasso', params=lasso_alphas, x_data_n=x_data, index_n=x_data.columns, y_target_n=y_target)

print("\n")

sort_column = 'alpha : ' + str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column, ascending=False)
print("coeff_lasso_df \n", coeff_lasso_df)

print("\n")

print("Elastic Net \n")

elastic_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df = get_linear_reg_eval('ElasticNet', params=elastic_alphas, x_data_n=x_data, index_n=x_data.columns, y_target_n=y_target)
sort_column = 'alpha : ' + str(elastic_alphas[0])
coeff_elastic_df.sort_values(by=sort_column, ascending=False)

print("\n")
print("coeff_elastic_df \n", coeff_elastic_df)

print("\n")
print("선형 회귀 모델을 위한 데이터 변환 \n")

def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard' : scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax' : scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log' : scaled_data = np.log1p(input_data)
    else : scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, include_bias=False).fit_transform(scaled_data)
    return scaled_data

alphas = [0.1, 1, 10, 100]
scale_methods = [(None, None), ('Standard', None), ('Standard', 2), ('MinMax', None), ('Log', None)]

for scale_method in scale_methods:
    x_data_scaled = get_scaled_data(method=scale_method[0], p_degree=scale_method[1], input_data=x_data)
    print(scale_method[0], scale_method[1])
    get_linear_reg_eval('Ridge', params=alphas, x_data_n=x_data_scaled, y_target_n=y_target)
    print("\n")

