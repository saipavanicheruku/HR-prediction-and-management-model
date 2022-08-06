import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Final.functions_file_j_v2 import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from lightgbm.sklearn import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score, roc_auc_score, roc_curve, auc
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

train_df = pd.read_csv(r"C:/Users/pavan/PycharmProjects/pythonProject/MidTerm/aug_train.csv")
print(train_df.head())
#print(train_df.shape)
train_df.info()

# unecessary column
train_df.drop('enrollee_id', axis=1, inplace=True)
print(train_df.isnull().sum(axis=0) / len(train_df.isnull()))  # number of null values

# dictionary
gender_dict = {'Other': 0, 'Male': 1, 'Female': 2}
enrolled_univeristy_dict = {'no_enrollment': 0, 'Full time course': 1, 'Part time course': 2}
education_level_dict = {'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 4}
major_discipline_dict = {'STEM': 0, 'Business Degree': 1, 'Arts': 2, 'Humanities': 3, 'No Major': 4, 'Other': 5}
experience_dict = {'<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
                   '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '>20': 21}
company_size_dict = {'<10': 0, '10/49': 1, '100-500': 2, '1000-4999': 3, '10000+': 4, '50-99': 5, '500-999': 6,
                     '5000-9999': 7}
company_type_dict = {'Pvt Ltd': 0, 'Funded Startup': 1, 'Early Stage Startup': 2, 'Other': 3, 'Public Sector': 4,
                     'NGO': 5}
last_new_job_dict = {'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5}
relevent_experience_dict = {'No relevent experience': 0, 'Has relevent experience': 1}


train_df.loc[:, 'gender'] = train_df['gender'].map(gender_dict)
train_df.loc[:, 'enrolled_university'] = train_df['enrolled_university'].map(enrolled_univeristy_dict)
train_df.loc[:, 'education_level'] = train_df['education_level'].map(education_level_dict)
train_df.loc[:, 'major_discipline'] = train_df['major_discipline'].map(major_discipline_dict)
train_df.loc[:, 'experience'] = train_df['experience'].map(experience_dict)
train_df.loc[:, 'company_size'] = train_df['company_size'].map(company_size_dict)
train_df.loc[:, 'company_type'] = train_df['company_type'].map(company_type_dict)
train_df.loc[:, 'last_new_job'] = train_df['last_new_job'].map(last_new_job_dict)
train_df.loc[:, 'relevent_experience'] = train_df['relevent_experience'].map(relevent_experience_dict)
train_df.loc[:, 'city'] = LabelEncoder().fit_transform(train_df.loc[:, 'city'])

#train_df.head()

#plotting the null values before fixing NAN values
null_bar(train_df)

# replace null values with mode values
impute_mode(train_df, 'gender')
impute_mode(train_df, 'enrolled_university')
impute_mode(train_df, 'education_level')
impute_mode(train_df, 'major_discipline')
impute_mode(train_df, 'experience')
impute_mode(train_df, 'company_size')
impute_mode(train_df, 'company_type')
impute_mode(train_df, 'last_new_job')
print(train_df.isnull().sum(axis=0))

#plotting the null values after fixing NAN values
null_bar(train_df)

#descriptive analysis
#exploring some of the coulumns, distribution with respect to target column

#1 - city_development_index
kdeplotgraph(train_df, 'city_development_index')

#2 - experience
kdeplotgraph(train_df, 'experience')

# Imbalanced data for target column
countplotgraph(train_df, 'target')

# overfitting the imbalanced data
sm = SMOTE(sampling_strategy='minority', random_state=7)

# fitting the model and plotting the newly over fitted data
oversampled_trainX, oversampled_trainY = sm.fit_resample(train_df.drop('target', axis=1), train_df['target'])
oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)

# Balanced data for target column
countplotgraph(oversampled_train, 'target')

# correlation matrix
corrmatrix(oversampled_train)

#Splitting the data into train and test
y = oversampled_train['target']
X = oversampled_train.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# Standardizing the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

####### new models
# Model 1 - Random Forest Classifier
rf1 = RandomForestClassifier(random_state = 0)
rf1.fit(X_train_std, y_train)
#predicting test set
y_rf1 = rf1.predict(X_test_std)
print('Random Forest Classifer accuracy score:', "{:.0%}".format(accuracy_score(y_test, y_rf1)))
prediction_test_rf1 = scores(rf1, X_test_std, y_test)
cmatrix(y_test, prediction_test_rf1)

# Model 2 - LGBM Classifier
lgbm = LGBMClassifier()
lgbm.fit(X_train_std, y_train)
#predicting test set
y_lgbm = lgbm.predict(X_test_std)
print('LGBM accuracy score:', "{:.0%}".format(accuracy_score(y_test, y_lgbm)))
prediction_test_lgbm = scores(lgbm, X_test_std, y_test)
cmatrix(y_test, prediction_test_lgbm)

# Model 3 - GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_std, y_train)
#predicting test set
y_gnb = gnb.predict(X_test_std)
print('GaussinNB accuracy score:',"{:.0%}".format(accuracy_score(y_test,y_gnb)))
prediction_test_gnb = scores(gnb, X_test_std, y_test)
cmatrix(y_test, prediction_test_gnb)

# Model 4 - EasyEnsemble Classifier
easy_lgbm = EasyEnsembleClassifier(base_estimator= LGBMClassifier(random_state=42), n_estimators=250, n_jobs=1,
                       random_state=42, replacement=True,
                       sampling_strategy='auto', verbose=0,
                       warm_start=True)
easy_lgbm.fit(X_train_std, y_train)
#predicting test set
y_elgbm = easy_lgbm.predict(X_test_std)
print('easylgbm accuracy score:',"{:.0%}".format(accuracy_score(y_test,y_elgbm)))
prediction_test_easylgbm = scores(easy_lgbm, X_test_std, y_test)
cmatrix(y_test, prediction_test_easylgbm)

## Mid term model
# Model 5 - Linear SVC
lsvc = LinearSVC(max_iter=5000)
lsvc.fit(X_train_std, y_train)
print('LSVC accuracy score test before feature elimination:', "{:.0%}".format(accuracy_score(y_test, lsvc.predict(X_test_std))))
# getting the precision, recall, and F1 scores before RFE
prediction_test_lsvc = scores(lsvc, X_test_std, y_test)
# Confusion matrix before feature elimination
cmatrix(y_test, prediction_test_lsvc)

# Recursive feature elimination
rfe = RFE(estimator=lsvc, n_features_to_select=7, verbose=1)
rfe.fit(X_train_std, y_train)
# Features selected and kept
df_features = X.columns.to_frame(index=False,name='Feature')
df_features["Elimination"] = pd.Series(rfe.ranking_, index=df_features.index)
df_features.sort_values('Elimination', ascending=False, inplace=True)
print(df_features)

# getting the precision, recall, and F1 scores before RFE
print('LSVC accuracy score test after feature elimination:', "{:.0%}".format(accuracy_score(y_test, rfe.predict(X_test_std))))
prediction_test_rfe = scores(rfe, X_test_std, y_test)
# Confusion matrix before feature elimination
cmatrix(y_test, prediction_test_rfe)

# Plotting the ROC curves for all the models tried
fig = plot_roc_curve(rf1, X_test_std, y_test)
fig = plot_roc_curve(lgbm, X_test_std, y_test, ax = fig.ax_)
fig = plot_roc_curve(lsvc, X_test_std, y_test, ax = fig.ax_)
fig = plot_roc_curve(gnb, X_test_std, y_test, ax = fig.ax_)
fig = plot_roc_curve(easy_lgbm, X_test_std, y_test, ax = fig.ax_)
fig.figure_.suptitle("ROC curve comparison")
plt.show()