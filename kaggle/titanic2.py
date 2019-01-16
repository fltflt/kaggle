#coding=utf-8
import pandas as pd
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import model_selection
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_validate
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_df=pd.read_csv('D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\train.csv')



### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values#age是否为空，并统计空的个数
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:,0]


    # X即特征属性值
    X = known_age[:,1:]


    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

train_df, rfr = set_missing_ages(train_df)
train_df= set_Cabin_type(train_df)


dummies_Cabin = pd.get_dummies(train_df['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(train_df['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(train_df['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(train_df['Pclass'], prefix= 'Pclass')

train_df1 = pd.concat([train_df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# #objs: series，dataframe或者是panel构成的序列lsit 
# axis： 需要合并链接的轴，0是行，1是列 
# join：连接的方式 inner，或者outer
train_df1.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)



scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(np.array(train_df1['Age']).reshape(-1,1))
train_df1['Age_scaled'] = scaler.fit_transform(np.array(train_df1['Age']).reshape(-1,1), age_scale_param)

#对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
fare_scale_param = scaler.fit(np.array(train_df1['Fare']).reshape(-1,1))
train_df1['Fare_scaled'] = scaler.fit_transform(np.array(train_df1['Fare']).reshape(-1,1), fare_scale_param)

# 用正则取出我们要的属性值

# fit到RandomForestRegressor之中





data_test=pd.read_csv('D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\test.csv')

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
predicted_Ages = null_age[:, 1:]
predictedAges = rfr.predict(predicted_Ages)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')#转换成多个二进制值
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(np.array(df_test['Age']).reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(np.array(df_test['Fare']).reshape(-1,1), fare_scale_param)

train_df2 = train_df1.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df2.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# cv_results = cross_validate(model, X, y, cv=5, return_train_score=False)
# print (cv_results['test_score'])
param_grid = {'random_state': [20 ,50 ,30,40], 'C': [1, 0.8, 0.7,0.6],'tol':[1e-6,1e-4,1e-7,1e-3]}
grid=model_selection.GridSearchCV(linear_model.LogisticRegression(penalty='l1'),param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True,cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise',return_train_score=True)
grid=grid.fit(X, y)


# model=linear_model.LogisticRegression(penalty='l1',C=1, random_state=20, tol= 1e-06)
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# model.fit(X, y)
# predictions =model.predict(test)
predictions=grid.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\logistic_regression_predictions5.csv", index=False)

# model = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = train_df1.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.values[:,1:]
# y = all_data.values[:,0]

# # 分割数据，按照 训练数据:cv数据 = 7:3的比例
# split_train, split_cv = cross_validation.train_test_split(train_df1, test_size=0.3, random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # 生成模型
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(train_df.values[:,1:], train_df.values[:,0])

# # 对cross validation数据进行预测

# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.values[:,1:])

# origin_data_train = pd.read_csv('D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\train.csv')
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]

# train_df = train_df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

# train_np = train_df.values

# # y即Survival结果
# y = train_np[:, 0]

# # X即特征属性值
# X = train_np[:, 1:]

# # fit到BaggingRegressor之中
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
# bagging_clf.fit(X, y)



# test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
# predictions = bagging_clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
# result.to_csv("D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\logistic_regression_bagging_predictions.csv", index=False)
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, u"学习曲线", X, y)






# print (cross_validation.cross_val_score(model, X, y, cv=5))


# print (pd.DataFrame({"columns":list(train_df2.columns)[1:], "coef":list(model.coef_.T)}))

# print (train_df.describe())

# print (train_df.info())



# # # print(train_df.describe() )
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# fig = plt.figure()
# fig.set(alpha=0.5)  # 设定图表颜色alpha参数

# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# train_df.Survived.value_counts().plot(kind='bar')# 柱状图 
# plt.title("获救情况 (1为获救)") # 标题
# plt.ylabel("人数")

# plt.subplot2grid((2,3),(0,1))
# train_df.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")

# plt.subplot2grid((2,3),(0,2))
# plt.scatter(train_df.Survived, train_df.Age)
# plt.ylabel(u"年龄")                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y') 
# plt.title(u"按年龄看获救分布 (1为获救)")


# plt.subplot2grid((2,3),(1,0), colspan=2)
# train_df.Age[train_df.Pclass == 1].plot(kind='kde')   
# train_df.Age[train_df.Pclass == 2].plot(kind='kde')
# train_df.Age[train_df.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度") 
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


# plt.subplot2grid((2,3),(1,2))
# train_df.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")

# Survived_0 = train_df.Pclass[train_df.Survived == 0].value_counts()
# Survived_1 = train_df.Pclass[train_df.Survived == 1].value_counts()
# df=pd.DataFrame({"获救":Survived_1, "未获救":Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级") 
# plt.ylabel(u"人数") 

# Survived_m = train_df.Survived[train_df.Sex == 'male'].value_counts()
# Survived_f = train_df.Survived[train_df.Sex == 'female'].value_counts()
# df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
# df.plot(kind='bar', stacked=True)#柱状图，累积着的柱状图
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"性别") 
# plt.ylabel(u"人数")
# plt.show()

# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况")

# ax1=fig.add_subplot(141)
# train_df.Survived[train_df.Sex == 'female'][train_df.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
# ax1.legend([u"女性/高级舱"], loc='best')

# ax2=fig.add_subplot(142, sharey=ax1)
# train_df.Survived[train_df.Sex == 'female'][train_df.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"女性/低级舱"], loc='best')

# ax3=fig.add_subplot(143, sharey=ax1)
# train_df.Survived[train_df.Sex == 'male'][train_df.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/高级舱"], loc='best')

# ax4=fig.add_subplot(144, sharey=ax1)
# train_df.Survived[train_df.Sex == 'male'][train_df.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/低级舱"], loc='best')

# plt.show()

# g = train_df.groupby(['SibSp','Survived'])
# df1 = pd.DataFrame(g.count()['PassengerId'])
# print (df1)

# g = train_df.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print (df)