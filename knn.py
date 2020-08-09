import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def plot_accuracy(x_train, x_test, y_train, y_test):
    """Computes and plots the accuracy rate over k iterations"""
    accuracy = []
    k_values = [3,5,7,9,11,13,15,17,19,21]
    
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)
        pred_k = knn_classifier.predict(x_test)
        accuracy.append(accuracy_score(pred_k, y_test))
    
    print(accuracy)
    
    plt.plot(k_values, accuracy, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize='10')
    plt.title('Accuracy vs. K for Churn')
    plt.ylabel('Number of neighbors: k')
    plt.ylabel('Accuracy (%)')
    plt.show()

def knn_cm(x_train, x_test, y_train, y_test, neighbors):
    """Computes a confusion matrix for KNN with a user entered number"""
    knn_classifier = KNeighborsClassifier(n_neighbors=neighbors)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    tp = cm[0][0] # True positive

    fn = cm[1][0] # False Negative

    fp = cm[0][1] # False Positive

    tn = cm[1][1] # True negative

    tpr = tp/(tp+fn) # True Positive Rate
    tnr = tn/(tn + fp) # True negative Rate

    print("===================================")
    print('Confusion Matrix')
    print("===================================")
    print(cm)
    print("===================================")
    print('True Positives - ' + str(tp))
    print('False Negatives - ' + str(fn))
    print('False Positives - ' + str(fp))
    print('True Negatives - ' + str(tn))
    print("===================================")
    print('True Positive Rate - ' + str(round(tpr,3)))
    print('True Negative Rate - ' + str(round(tnr,3)))