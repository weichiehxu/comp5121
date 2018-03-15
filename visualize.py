import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from preprocess import Rename


#Draw the Heatmap
def plot0():
    plt.figure(figsize=(9, 4))
    corr = Rename.data_set.corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r',
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                center=0, linewidths=0.5)
    plt.title('Heatmap of Correlation Matrix')
    plt.savefig("Heatmap.png")

#Draw the K-Means Clustering
def plot1():
    plt.figure(figsize=(12, 8))

    # Employee who left
    print("--------Let's find some interests-------\n"
          "Print the employee who left\n")
    plt.subplot(1, 2, 1)
    plt.plot(Rename.data_set.satisfaction[Rename.data_set.left == 1],
             Rename.data_set.evaluation[Rename.data_set.left == 1],
             'ro', alpha=0.2)
    plt.ylabel('Last Evaluation')
    plt.title('Employees who left')
    plt.xlabel('Satisfaction level')

    # Employees who stayed
    print("Print the employee who stayed\n")
    plt.subplot(1, 2, 2)
    plt.title('Employees who stayed')
    plt.plot(Rename.data_set.satisfaction[Rename.data_set.left == 0],
             Rename.data_set.evaluation[Rename.data_set.left == 0], 'bo', alpha=0.2)
    plt.xlim([0.4, 1])
    plt.ylabel('Last Evaluation')
    plt.xlabel('Satisfaction level')
    plt.savefig('employee.png')

#Draw the K-Means Clustering 2
def plot2():
    kmeans_df = Rename.data_set[Rename.data_set.left == 1].drop([u'project',
                                                                 u'hours', u'years', u'accident',
                                                                 u'left', u'promotion', u'department', u'salary'], axis=1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(kmeans_df)
    print(kmeans.cluster_centers_)

    left = Rename.data_set[Rename.data_set.left == 1]
    left.label = kmeans.labels_
    plt.figure(figsize=(10, 7))
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Last Evaluation score')
    plt.title('Clustering those who left')
    plt.plot(left.satisfaction[left.label == 0], left.evaluation[left.label == 0], 'o', alpha=0.2,
             color='red')
    plt.plot(left.satisfaction[left.label == 1], left.evaluation[left.label == 1], 'o', alpha=0.2,
             color='green')
    plt.plot(left.satisfaction[left.label == 2], left.evaluation[left.label == 2], 'o', alpha=0.2,
             color='blue')
    plt.savefig('clustering.png')
    print("\n---------End---------\n")

#Draw random forest bar chart
def plot3():
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.figure(figsize=(12, 7))

    # Convert these variables into categorical variables
    Rename.data_set["department"] = Rename.data_set["department"].astype('category').cat.codes
    Rename.data_set["salary"] = Rename.data_set["salary"].astype('category').cat.codes

    # Create train and test splits
    y = Rename.data_set['left']
    X = Rename.data_set.drop(['left'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)

    dtree = RandomForestClassifier()
    dtree.fit(x_train, y_train)
    print(dtree.feature_importances_)

    ''' plot the importances '''
    importances = dtree.feature_importances_
    feat_names = x_train.columns
    indices = np.argsort(importances)[::-1]
    feat_importance = importances[indices]
    feat_cols = feat_names[indices]
    df_feature_importance = pd.DataFrame({'Features': feat_cols, 'feature_importance': feat_importance})
    sns.barplot(x='Features', y='feature_importance', data=df_feature_importance)
    plt.savefig('randomforest.png')
