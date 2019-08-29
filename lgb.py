from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import csv
import lightgbm as lgb
# print('s')
# dir1="train_all.csv"
# dir2="new_train.csv"
# dir3="new_y.csv"
# a=1
# desX = np.loadtxt(open(dir1, "rb"), delimiter=",", skiprows=0,
#                   usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24))
# desY = np.loadtxt("train_all.csv", delimiter=",", skiprows=0, usecols=(25))


def turning():

    train_data = np.array(pd.read_csv("train_all.csv"))
    predict_data = np.array(pd.read_csv("republish_test.csv"))
    desX = train_data[:, :25]
    desY = train_data[:, 25]
    x_predict = predict_data[:, :25]
    user_id = predict_data[:, 25]

    # x_train, x_test, y_train, y_test = train_test_split(desX, desY, test_size=0.2, random_state=0)
    # print(y_test)
    # print(len(desY))

    service_num_dict = {}
    label = []

    transfer_number = 0

    for service_num in desY:
        if service_num not in service_num_dict.keys():
            service_num_dict[service_num] = transfer_number
            transfer_number += 1
        label.append(service_num_dict[service_num])

# clf=KNeighborsClassifier();
#
# clf.fit(desX,desY.ravel())
# score = clf.score(x_test,y_test.ravel())
# pre_label=clf.predict(x_test)
# print('the score is :', score)
# print(confusion_matrix(y_test, pre_label))
# print(classification_report(y_test, pre_label))
# joblib.dump(clf,"LGB.pkl")


    # print("LGB test")
    params = {
        'boosting_type': 'gbdt',
        # 'objective': 'regression',
        'objective': 'multiclass',
        'num_class': 11,
        'learning_rate': 0.1,
        'num_leaves': 116,
        'max_depth': 9,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        }

    # data_train = lgb.Dataset(desX, label=label)
    # cv_results = lgb.cv(
    #     params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='multi_error',
    #     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
    # print(cv_results)
    # print('best n_estimators:', len(cv_results['multi_error-mean']))
    # print('best cv score:', cv_results['multi_error-mean'][-1])

    # model_lgb = lgb.LGBMClassifier(objective='multiclass', num_class=11, num_leaves=116,
    #                               learning_rate=0.1, n_estimators=1000, max_depth=9,
    #                               metric='multi_logloss', bagging_fraction=0.8, feature_fraction=0.8)
    #
    # params_test1={
    #     # 'max_depth': range(7, 12, 2),
    #     # 'num_leaves': [183, 191, 210, 125]
    #     # 'min_child_samples': [20, 21, 22, 23, 24], 'min_child_weight': [0.001, 0.002]
    #     'feature_fraction': [0.8, 0.9, 1.0],    'bagging_fraction': [0.8, 0.9, 1.0]
    # }
    # gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, verbose=1, n_jobs=4)
    #
    # gsearch1.fit(desX, desY.astype('int'))
    # print(gsearch1.best_params_, gsearch1.best_score_, gsearch1.grid_scores_)



    #
    clf = lgb.LGBMClassifier(boosting_type="gbdt", objective='multiclass', num_class=11, num_leaves=116,
                                      learning_rate=0.05, n_estimators=10000, max_depth=9,
                                      metric='multi_logloss', min_child_samples=22, feature_fraction=0.9,
                                      bagging_fraction=1.0, min_child_weight=0.001,
                                      reg_alpha=0.0, reg_lambda=0.5)
    # clf = lgb.LGBMClassifier()
    clf.fit(desX, desY.astype('int'))
    # pre_label=clf.predict(x_test)
    # print(pre_label)
    joblib.dump(clf, "lightgbm_2018_10_21_1.pkl")

    # clf = joblib.load("lightgbm_2018_10_18.pkl")
    # score = clf.score(desX, desY.astype('int'))
    predict = clf.predict(x_predict)
    # print('the score is :', score)
    f = open("submission_lgb_2018_10_21_1.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["user_id", "current_service"])
    for i in range(0, len(user_id)):
        writer.writerow([user_id[i], predict[i]])
        # if i != 131428 or i != 170278:
        #     writer.writerow([user_id[i], y_predict[i]])
        # else:
        #     writer.writerow([user_id[i], random.choice(y_train)])
    f.close()
# print(confusion_matrix(y_test, pre_label))
# print(classification_report(y_test, pre_label))


if __name__ == '__main__':

    turning()