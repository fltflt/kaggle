from sklearn import linear_model
import pandas as pd
import numpy as np 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import model_selection
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
#首先引入需要的库和函数
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import BaggingClassifier


#数据处理流程
# 填补age缺失值
# 定性数据性别 转为定量数据0,1
# name先不管
# plass转为独热编码
# Embarked分列，独热编码

##1 将数据集合并
train_data_org=pd.read_csv('D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\train.csv')
test_data_org=pd.read_csv('D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\test.csv')
test_data_org['Survived'] = 0
combined_train_test = train_data_org.append(test_data_org)
# print (combined_train_test.describe())
# print (combined_train_test.info())
# print (train_data_org['Survived'].value_counts())
# # 相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.
# print (train_data_org.drop('PassengerId',axis=1).corr())
# print (train_data_org.columns.values)
# print (train_data_org.head())
# #统计各个变量对生存的影响
# print (train_data_org[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# print (train_data_org[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# print (train_data_org[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# print (train_data_org.groupby(['Sex'])['Sex','Survived'].mean())

# #2 对sex,pclass,Parch,sibsp进行处理 ，直接分列，pd.concat增加新列进去
# print (combined_train_test.groupby(by=["Pclass","Embarked"]).Fare.mean())
combined_train_test["Fare"].fillna(14.435422,inplace=True)##用pclass=3和Embarked=S的平均数14.435422来填充

combined_train_test = pd.get_dummies(combined_train_test,columns=['Sex'])
combined_train_test = pd.get_dummies(combined_train_test,columns=['Pclass'])

combined_train_test['SibSp_Parch'] = combined_train_test['SibSp'] + combined_train_test['Parch']
combined_train_test = pd.get_dummies(combined_train_test,columns = ['SibSp','Parch','SibSp_Parch']) 


#3 对Embarked处理，先填充缺失值，对缺失的Embarked以众数来填补，返回的众数可能有很多行，

#iloc[0]表示最多的那个，即第一行，再将Embarked的三个上船港口分为3列，每一列均只包含0和1两个值

if combined_train_test['Embarked'].isnull().sum() != 0:
	combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
combined_train_test = pd.get_dummies(combined_train_test,columns=['Embarked'])

#4 对name处理,map(title_Dict)就是对数据集中的每一个进行遍历，然映射为title_Dict中的值,从名字中提取出称呼： df['Name].str.extract()是提取函数,配合正则一起使用

combined_train_test['Name']=combined_train_test['Name'].str.extract('.+,(.+)')
combined_train_test['Name']=combined_train_test['Name'].str.extract( '^(.+?)\.')
combined_train_test['Name']=combined_train_test['Name'].str.strip()
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master'], 'Master'))
combined_train_test['Name'] = combined_train_test['Name'].map(title_Dict)
#对姓名进行分列
combined_train_test = pd.get_dummies(combined_train_test,columns=['Name'])


#5 对Fare处理,该特征有缺失值,先找出缺失值的那调数据,然后用平均数填充

if combined_train_test['Fare'].isnull().sum() != 0:
        combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform('mean'))



combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']


#6对ticket处理，
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isnumeric() else x)
combined_train_test.drop('Ticket',inplace=True,axis=1)
combined_train_test = pd.get_dummies(combined_train_test,columns=['Ticket_Letter'],drop_first=True)



#7对fare处理
def fare_category(fare):
        if fare <= 4:
            return 0
        elif fare <= 10:
            return 1
        elif fare <= 30:
            return 2
        elif fare <= 45:
            return 3
        else:
            return 4
combined_train_test['Fare_Category'] = combined_train_test['Fare'].map(fare_category)
combined_train_test = pd.get_dummies(combined_train_test,columns=['Fare_Category'])


#8 对年龄处理
### 使用 RandomForestClassifier 填补缺失的年龄属性



# age_df = combined_train_test.filter(regex='Age|SibSp_*|Parch_*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
# print (age_df)
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = combined_train_test.filter(regex='Age|SibSp_*|Parch_*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')

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

combined_train_test, rfr = set_missing_ages(combined_train_test)



combined_train_test= set_Cabin_type(combined_train_test)
combined_train_test = pd.get_dummies(combined_train_test,columns=['Cabin'])

combined_train_test.drop(['Group_Ticket','Fare'], axis=1, inplace=True)
# print (combined_train_test.columns)
# print (combined_train_test.info())

# 数据处理完之后，划分数据集
from sklearn.model_selection._validation import cross_val_score
from sklearn.model_selection import train_test_split
train_data = combined_train_test[:891]
test_data = combined_train_test[891:]
titanic_train_data_X = train_data.drop(['Survived'],axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'],axis=1)




# cross_validation(SVC(gamma=0.15, C=1),CV=5)

# alg1=DecisionTreeClassifier(min_samples_split=2, max_depth=1).fit(titanic_train_data_X,titanic_train_data_Y)
# alg2=SVC(gamma=0.15, C=1).fit(titanic_train_data_X,titanic_train_data_Y)  #由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
# alg3=RandomForestClassifier(n_estimators=180,min_samples_split=4, max_depth=8).fit(titanic_train_data_X,titanic_train_data_Y)
# alg4=AdaBoostClassifier(learning_rate=0.5,n_estimators= 100).fit(titanic_train_data_X,titanic_train_data_Y)
# alg5=KNeighborsClassifier(n_neighbors= 9, leaf_size= 30).fit(titanic_train_data_X,titanic_train_data_Y)
# alg6=XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1,colsample_bytree=0.5, subsample= 0.6).fit(titanic_train_data_X,titanic_train_data_Y)
# alg7=GradientBoostingClassifier(learning_rate=1.0, n_estimators=6, subsample=0.6).fit(titanic_train_data_X,titanic_train_data_Y)

# def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):

#     # random forest
#     rf_est = RandomForestClassifier(random_state=0)
#     rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
#     rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
#     rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
#     print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
#     print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
#     print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
#     feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
#                                           'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
#     print('Sample 10 Features from RF Classifier')
#     print(str(features_top_n_rf[:10]))

#     # AdaBoost
#     ada_est =AdaBoostClassifier(random_state=0)
#     ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
#     ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
#     ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
#     print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
#     print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
#     print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
#     feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
#                                            'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
#     print('Sample 10 Feature from Ada Classifier:')
#     print(str(features_top_n_ada[:10]))

#     # # ExtraTree
#     # et_est = ExtraTreesClassifier(random_state=0)
#     # et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
#     # et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
#     # et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
#     # print('Top N Features Best ET Params:' + str(et_grid.best_params_))
#     # print('Top N Features Best ET Score:' + str(et_grid.best_score_))
#     # print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
#     # feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
#     #                                       'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     # features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
#     # print('Sample 10 Features from ET Classifier:')
#     # print(str(features_top_n_et[:10]))

#     # GradientBoosting
#     gb_est =GradientBoostingClassifier(random_state=0)
#     gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
#     gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
#     gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
#     print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
#     print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
#     print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
#     feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
#                                            'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
#     print('Sample 10 Feature from GB Classifier:')
#     print(str(features_top_n_gb[:10]))

#     # DecisionTree
#     dt_est = DecisionTreeClassifier(random_state=0)
#     dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
#     dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
#     dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
#     print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
#     print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
#     print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
#     feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
#                                           'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
#     print('Sample 10 Features from DT Classifier:')
#     print(str(features_top_n_dt[:10]))

#     # merge the three models
#     features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 
#                                ignore_index=True).drop_duplicates()

#     features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 
#                                    feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)

#     return features_top_n , features_importance


# get_top_n_features(titanic_train_data_X, titanic_train_data_Y, 30)


# print (titanic_train_data_X.columns)
# GBDT作为基模型的特征选择,titanic_train_data_Y
# feature=SelectFromModel(GradientBoostingClassifier()).fit_transform(titanic_train_data_X,titanic_train_data_Y)
# print (feature)
# fit_
# b = [i[1] for i in feature]
# print (b)


# logreg = LogisticRegression()
# logreg.fit(titanic_train_data_X,titanic_train_data_Y)
# Y_pred1 = logreg.predict(titanic_test_data_X)
# acc_log = round(logreg.score(titanic_train_data_X,titanic_train_data_Y) * 100, 2)
# print (logreg.score(titanic_test_data_X,Y_pred1))
# print (acc_log)

# svc = SVC(probability=True,random_state=29,C=5, gamma= 0.05).fit(titanic_train_data_X,titanic_train_data_Y)
# Y_pred2 = svc.predict(titanic_test_data_X)
# acc_svc = round(svc.score(titanic_train_data_X,titanic_train_data_Y)*100, 2)
# print (svc.score(titanic_test_data_X,Y_pred2))
# print (acc_svc)

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
#                         train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
#     """
#     画出data在某模型上的learning curve.
#     参数解释
#     ----------
#     estimator : 你用的分类器。
#     title : 表格的标题。
#     X : 输入的feature，numpy类型
#     y : 输入的target vector
#     ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
#     cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
#     n_jobs : 并行的的任务数(默认1)
#     """
#     train_sizes, train_scores, test_scores = learning_curve(
#         svc, titanic_train_data_X,titanic_train_data_Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)

#     if plot:
#         plt.figure()
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel(u"训练样本数")
#         plt.ylabel(u"得分")
#         plt.gca().invert_yaxis()
#         plt.grid()

#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
#                          alpha=0.1, color="b")
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
#                          alpha=0.1, color="r")
#         plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
#         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

#         plt.legend(loc="best")

#         plt.draw()
#         plt.show()
#         plt.gca().invert_yaxis()

#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     return midpoint, diff

# plot_learning_curve(svc, u"学习曲线", titanic_train_data_X,titanic_train_data_Y)


# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(titanic_train_data_X,titanic_train_data_Y)
# Y_pred3 = decision_tree.predict(titanic_test_data_X)
# acc_decision_tree = round(decision_tree.score(titanic_train_data_X,titanic_train_data_Y) * 100, 2)
# print (decision_tree.score(titanic_test_data_X,Y_pred3))
# print (acc_decision_tree)


# result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':Y_pred2.astype(np.int32)})
# result.to_csv("D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\res9.csv",index=False)
# 模型构建，预测最佳参数
# def fit_model(alg,parameters):
# 	X= titanic_train_data_X
# 	y= titanic_train_data_Y
# 	scorer=make_scorer(roc_auc_score)
# 	grid = GridSearchCV(alg,parameters,scoring=scorer,cv=5)
# 	grid = grid.fit(X,y)
# 	print (grid.best_params_)
# 	print (grid.best_score_)
# 	return grid

# # 列出需要使用的算法
# alg1=DecisionTreeClassifier(min_samples_split=2, max_depth=1).fit(titanic_train_data_X,titanic_train_data_Y)
# alg2=SVC(gamma=0.15, C=1).fit(titanic_train_data_X,titanic_train_data_Y)  #由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
# alg3=RandomForestClassifier(n_estimators=180,min_samples_split=4, max_depth=8).fit(titanic_train_data_X,titanic_train_data_Y)
# alg4=AdaBoostClassifier(learning_rate=0.5,n_estimators= 100).fit(titanic_train_data_X,titanic_train_data_Y)
# alg5=KNeighborsClassifier(n_neighbors= 9, leaf_size= 30).fit(titanic_train_data_X,titanic_train_data_Y)
# alg6=XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1,colsample_bytree=0.5, subsample= 0.6).fit(titanic_train_data_X,titanic_train_data_Y)
# alg7=GradientBoostingClassifier(learning_rate=1.0, n_estimators=6, subsample=0.6).fit(titanic_train_data_X,titanic_train_data_Y)

# # # 列出需要调整的参数范围
# # parameters1={'max_depth':range(1,100),'min_samples_split':range(2,30)}
# # parameters2 = {"C":range(1,20), "gamma": [0.05,0.1,0.15,0.2,0.25]}
# # parameters3_1 = {'n_estimators':range(10,200,10)}
# # parameters3_2 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}  #搜索空间太大，分两次调整参数
# # parameters4 = {'n_estimators':range(10,200,10),'learning_rate':[i/10.0 for i in range(5,15)]}
# # parameters5 = {'n_neighbors':range(2,10),'leaf_size':range(10,80,20)  }
# # parameters6_1 = {'n_estimators':range(10,200,10)}
# # parameters6_2 = {'max_depth':range(1,10),'min_child_weight':range(1,10)}
# # parameters6_3 = {'subsample':[0.5,0.6,0.7,0.8,0.9], 'colsample_bytree':[0.5,0.6,0.7,0.8,0.9]}#搜索空间太大，分三次调整参数
# # parameters7= {'n_estimators':range(1,10),'learning_rate':[i/10.0 for i in range(5,15)],'subsample':[0.5,0.6,0.7,0.8,0.9,1]}
# # clf1=fit_model(alg1,parameters1)

# # clf2=fit_model(alg2,parameters2)

# # clf3_m1=fit_model(alg3,parameters3_1)

# # # # # {'max_depth': 8, 'min_samples_split': 9}
# # # # # {'C': 5, 'gamma': 0.05}
# # # # # {'n_estimators': 60}

# # # # # alg3=RandomForestClassifier(random_state=29,n_estimators=180)
# # clf3=fit_model(alg3,parameters3_2)

# # clf4=fit_model(alg4,parameters4)

# # clf5=fit_model(alg5,parameters5)

# # # # # {'min_samples_split': 7, 'max_depth': 9}
# # # # # 0.7749933003569167
# # # # # {'learning_rate': 0.5, 'n_estimators': 100}
# # # # # 0.7917113784351295
# # # # # {'n_neighbors': 9, 'leaf_size': 30}
# # # # # 0.7763078166938742

# # clf6_m1=fit_model(alg6,parameters6_1)
# # # # # alg6=XGBClassifier(n_estimators=140,random_state=29,n_jobs=-1)
# # clf6_m2=fit_model(alg6,parameters6_2)


# # # # # {'colsample_bytree': 0.5, 'subsample': 0.6}
# # # # # 0.8053253440925475
# # # # alg6=XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1)
# # clf6=fit_model(alg6,parameters6_3)
# # clf7=fit_model(alg7,parameters7)



# # # def save1(alg,i):
# # # 	test = titanic_test_data_X.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # # 	pred=alg.predict(test)
# # # 	result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':pred.astype(np.int32)})
# # # 	print (alg.score(test, pred))
# # # 	result.to_csv("D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\res_tan_{}.csv".format(i), index=False)

# # # i=1
# # # for alg in [alg1,alg2,alg3,alg4,alg5,alg6]:
# # #     save1(alg,i)
# # # #     i=i+1

# model=ensemble.VotingClassifier([('11',alg1),('22',alg2),('33',alg3),('44',alg4),('55',alg5),('66',alg6),('77',alg7)], voting='soft', weights=None, n_jobs=None, flatten_transform=True)
# model1=model.fit(titanic_train_data_X,titanic_train_data_Y)
# test = titanic_test_data_X
# for clf in (alg1,alg2,alg3,alg4,alg5,alg6,alg7):
#     clf1=clf.fit(titanic_train_data_X,titanic_train_data_Y)
#     clf1_pred=clf1.predict(test)
#     socre1=round(clf1.score(titanic_train_data_X,titanic_train_data_Y)*100, 2)
#     print (clf.__class__.__name__,socre1)
# socre2=round(model1.score(titanic_train_data_X,titanic_train_data_Y)*100, 2)
# print ('VotingClassifier',socre2)

# from sklearn.model_selection import cross_val_score
# bag_clf = BaggingClassifier(SVC(gamma=0.15, C=1), n_estimators=500,max_samples=100, bootstrap=True, n_jobs=-1)
# bag_clf.fit(titanic_train_data_X,titanic_train_data_Y)
# print (bag_clf.oob_score_)

# bag=bag_clf.fit(titanic_train_data_X,titanic_train_data_Y)
# bag_pred=bag.predict(titanic_test_data_X)
# bag_pred
# print (bag.oob_score_)

# DecisionTreeClassifier 78.23
# SVC 100.0
# RandomForestClassifier 89.0
# AdaBoostClassifier 85.19
# KNeighborsClassifier 64.42
# XGBClassifier 87.32
# VotingClassifier 89.34


# pred1=model1.predict(test)
# print (model1.score(test,pred1))
# acc_svc = round(model1.score(titanic_train_data_X,titanic_train_data_Y)*100, 2)
# print (acc_svc)
# result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':pred1.astype(np.int32)})
# result.to_csv("D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\res10.csv",index=False)

# # #对 Parch and SibSp 处理

# # def family_size_category(Family_Size):
# # 	if Family_Size<=1:
# # 		return 'Single'
# # 	elif Family_Size<=4:
# # 		return 'Small_Family'
# # 	elif Family_Size<=10:
# # 		return 'Large_Family'
# 	else:
# 		return 0

# combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
# combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

# le_family =preprocessing.LabelEncoder()
# le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
# fam_size_cat_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],prefix=combined_train_test[['Family_Size_Category']].columns[0])
# combined_train_test = pd.concat([combined_train_test, fam_size_cat_dummies_df], axis=1)

# def set_missing_ages(df):
# 	# 把已有的数值型特征取出来丢进Random Forest Regressor中
# 	age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# 	# 乘客分成已知年龄和未知年龄两部分
# 	known_age = age_df[age_df.Age.notnull()].values#age是否为空，并统计空的个数
# 	unknown_age = age_df[age_df.Age.isnull()].values
# 	# y即目标年龄
# 	y = known_age[:,0]

# 	# X即特征属性值
# 	X = known_age[:,1:]
# 	# fit到RandomForestRegressor之中
# 	rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
# 	rfr.fit(X, y)
# 	# 用得到的模型进行未知年龄结果预测
# 	predictedAges = rfr.predict(unknown_age[:, 1::])
# 	# 用得到的预测结果填补原缺失数据
# 	df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
# 	return df, rfr

# combined_train_test, rfr = set_missing_ages(combined_train_test)

# #对Ticket 处理

# combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
# combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isnumeric() else x)
# combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
# combined_train_test['Ticket_Number'].fillna(0,inplace=True)
# combined_train_test = pd.get_dummies(combined_train_test,columns=['Ticket','Ticket_Letter'])

# #对Cabin处理
# combined_train_test['Cabin_Letter'] = combined_train_test['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else x)
# combined_train_test = pd.get_dummies(combined_train_test,columns=['Cabin','Cabin_Letter'])

# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(np.array(combined_train_test['Age']).reshape(-1,1))
# combined_train_test['Age_scaled'] = scaler.fit_transform(np.array(combined_train_test['Age']).reshape(-1,1), age_scale_param)
# fare_scale_param = scaler.fit(np.array(combined_train_test['Fare']).reshape(-1,1))
# combined_train_test['Fare_scaled'] = scaler.fit_transform(np.array(combined_train_test['Fare']).reshape(-1,1), fare_scale_param)
# # combined_train_test.drop(['Embarked', 'Sex','Name', 'Parch', 'SibSp', 'Family_Size_Category'],axis=1,inplace=True)
# # combined_train_test.drop(['Small_Family'],axis=1,inplace=True)
# print(len(combined_train_test.columns))
# print (combined_train_test.describe())
# # from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# #递归特征消除法，返回特征选择后的数据
# #参数estimator为基模型
# #参数n_features_to_select为选择的特征个数
# target=combined_train_test.drop(['Survived'],axis=1)
# print (RFE(estimator=LogisticRegression(), n_features_to_select=5).fit_transform(combined_train_test, target))

# #将数据集分开
# train_data = combined_train_test[:891]
# test_data = combined_train_test[891:]

# titanic_train_data_Y = train_data['Survived']
# titanic_train_data_X = train_data.drop(['Survived'],axis=1)
# titanic_train_data_X=titanic_train_data_X.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# titanic_test_data_X = test_data.drop(['Survived'],axis=1)




# # result = pd.DataFrame({'PassengerId':titanic_test_data_X['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
# # result.to_csv("D:\\Sublime Text 3\\python\\Titanic Machine Learning from Disaster\\logistic_regression_predictions6.csv", index=False)





# # ('11',alg1),('22',alg2),('33',alg3),('44',alg4),('55',alg5),('66',alg6)


# # 
# # #建立模型
# # def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
# #         # 随机森林
# #         if __name__ == '__main__':
# # 	        rf_est = ensemble.RandomForestClassifier(random_state=42)
# # 	        rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
# # 	        rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
# # 	        rf_grid.fit(titanic_train_data_X,titanic_train_data_Y)
# # 	        #将feature按Importance排序
# # 	        feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
# # 	        features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
# # 	        print('Sample 25 Features from RF Classifier')
# # 	        print(str(features_top_n_rf[:25]))

# # 	        # AdaBoost
# # 	        ada_est = ensemble.AdaBoostClassifier(random_state=42)
# # 	        ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
# # 	        ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
# # 	        ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
# # 	        #排序
# # 	        feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),'importance': ada_grid.best_estimator_.feature_importances_}).sort_values( 'importance', ascending=False)
# # 	        features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']

# # 	        # ExtraTree
# # 	        et_est = ensemble.ExtraTreesClassifier(random_state=42)
# # 	        et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
# # 	        et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
# # 	        et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
# # 	        #排序
# # 	        feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
# # 	        features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
# # 	        print('Sample 25 Features from ET Classifier:')
# # 	        print(str(features_top_n_et[:25]))

# # 	        # 将三个模型挑选出来的前features_top_n_et合并
# # 	        features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et], ignore_index=True).drop_duplicates()

# # 	        return features_top_n_rf
# # feature_to_pick = 10
# # feature_top_n = get_top_n_features(titanic_train_data_X,titanic_train_data_Y,feature_to_pick)
# # titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
# # titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])





















# # print(combined_train_test.columns)输出索引
# # #对age 进行处理
# # missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Parch', 'Sex', 'SibSp', 'Family_Size', 'Family_Size_Category','Name', 'Fare', 'Fare_Category', 'Pclass', 'Embarked']])
# # print(missing_age_df)
# # missing_age_df = pd.get_dummies(missing_age_df,columns=['Name', 'Family_Size_Category', 'Fare_Category', 'Sex', 'Pclass' ,'Embarked'])
# # missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
# # missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]

# # def fill_missing_age(missing_age_train, missing_age_test):
# #  	missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
# #  	missing_age_Y_train = missing_age_train['Age']
# #  	missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
 	
# #  	gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
# #  	gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [3],'learning_rate': [0.01], 'max_features': [3]}
# #  	gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,  scoring='neg_mean_squared_error')
# #  	gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
# #  	print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
# #  	print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
# #  	print('GB Train Error for "Age" Feature Regressor:'+ str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
# #  	missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
# #  	print(missing_age_test['Age_GB'][:4])

# #  	lrf_reg = LinearRegression()
# #  	lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
# #  	lrf_reg_grid = model_selection.GridSearchCV(lrf_reg, lrf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
# #  	lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
# #  	print('Age feature Best LR Params:' + str(lrf_reg_grid.best_params_))
# #  	print('Age feature Best LR Score:' + str(lrf_reg_grid.best_score_))
# #  	print('LR Train Error for "Age" Feature Regressor' + str(lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
# #  	missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)
# #  	print(missing_age_test['Age_LRF'][:4])

# #  	print('shape1',missing_age_test['Age'].shape,missing_age_test[['Age_GB','Age_LRF']].mode(axis=1).shape)
# #  	# missing_age_test['Age'] = missing_age_test[['Age_GB','Age_LRF']].mode(axis=1)
# #  	missing_age_test['Age'] = np.mean([missing_age_test['Age_GB'],missing_age_test['Age_LRF']])
# #  	print(missing_age_test['Age'][:4])
# #  	drop_col_not_req(missing_age_test, ['Age_GB', 'Age_LRF'])
# #  	return missing_age_test
# # combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train,missing_age_test)
# # print (combined_train_test) 
# ##查看数据集基本信息
# # # print (test_data_org.info())
# # # print (train_data_org.info())
# # # print (test_data_org.describe())
# # print (test_data_org)