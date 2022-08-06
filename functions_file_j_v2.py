import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def null_bar(df):
    msno.bar(df, color="mediumslateblue", sort="ascending", fontsize=12);
    plt.tight_layout()
    plt.show()

def impute_mode(df, colname):
    df[colname] = df[colname].fillna((df[colname].mode()[0]))

def kdeplotgraph(df, colname):
    g = sns.kdeplot(df[colname][(df["target"] == 0) & (df[colname].notnull())], color="Blue", shade = True)
    g = sns.kdeplot(df[colname][(df["target"] == 1) & (df[colname].notnull())], ax =g, color="Red", shade= True)
    g.set_xlabel(colname)
    g.set_ylabel("Frequency")
    g.legend(["Not looking for job change,","looking for job change,"])
    plt.show()

def countplotgraph(df, colname):
    sns.countplot(x=colname, data=df, palette='PuRd')
    plt.show()

def corrmatrix(df):
    corr = df.corr()
    cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
    mask2 = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(df.corr(), mask=mask2, center=0, cmap=cmap, linewidth=1, annot=True, fmt='.2f')
    plt.show()

def scores(classifier,X_test,y_test):
    prediction_test = classifier.predict(X_test)
    precisionscore = round(precision_score(y_test, prediction_test), 3)
    print("Precision Score: ", precisionscore)
    recallscore = round(recall_score(y_test, prediction_test), 3)
    print("Recall Score: ", recallscore)
    f1score = round(f1_score(y_test, prediction_test), 3)
    print("F1 Score: ", f1score)
    return prediction_test

def cmatrix(y_test,prediction_test):
    cmatrix = confusion_matrix(y_test, prediction_test)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cmatrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cmatrix.flatten()/np.sum(cmatrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cmatrix, annot=labels, fmt='', cmap='Blues')
    plt.show()

#def evaluate(model, X_test, y_test):
#    y_pred = model.predict(X_test)
#    errors = abs(y_pred - y_test)
#    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#    print(classification_report(y_test,y_pred))
#    print(confusion_matrix(y_test,y_pred))
#    print('Recall Score = ',recall_score(y_test, y_pred))
#    print('Precision Score = ',precision_score(y_test, y_pred))
#    print('F1 score = ', f1_score(y_test,y_pred))
#    return evaluate