from preprocess import x_test, x_train, y_test, y_train
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB




def init():
    print("--------start data mining-------\n"
  
          "Naive Bayes, K-nn, Decision Tree and Random Forest will be shown:\n")


# Naive Bayes model
def bayes():
    naive_b = GaussianNB()
    naive_b.fit(x_train, y_train)
    bayes_result = naive_b.predict(x_test)
    print("The accuracy of Naive Bayes is: %f"
          % accuracy_score(y_test, bayes_result))


# K-nn model (The loop is to find the best k value)
def knn():
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        knn_result = knn.predict(x_test)
        scores.append(accuracy_score(y_test, knn_result))
    print("The maximum accuracy of K-nn is:", max(scores),
          "with k value = ", scores.index(max(scores)) + 1)
    '''plt.scatter(k_range, scores, c=scores, cmap=plt.cm.Blues)
    plt.title('Knn Testing Result')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('knn.png')'''
    '''If you want to show diagram, use: plt.show()'''
    '''If you want save diagram, use plt.savefig('knn.png')'''


# Decision tree
def decision_tree():
    dt = tree.DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    dt_result = dt.predict(x_test)
    print("The accuracy of decision tree is:", accuracy_score(y_test, dt_result))


# Random forest
def random_forest():
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    rf_result = rf.predict(x_test)
    print("The accuracy of random forest is: %f" %
          accuracy_score(y_test, rf_result))


def compare():
    print("Compared with accuracy of algorithms respectively,\n"
          "Random Forest has the highest accuracy value.\n"
          "Let's use a random data set to test with Random Forest:\n")


# Random test
def final_test():
    ft = RandomForestClassifier()
    ft.fit(x_train, y_train)
    a = random.randrange(0, 100)
    x_ft = x_test[a:a+5, :]
    predict_y = ft.predict(x_ft)
    for i in predict_y:
        print("Predict set is:", i)
    print("---------------------")
    for t in y_test[a:a+5]:
        print("Actual set is:", t)







