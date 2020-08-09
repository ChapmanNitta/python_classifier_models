import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def log_reg(x_train, y_train):
    """Builds a logistic regression classifier by taking in a dataframe
    , features, and class parameter"""

    log_reg_classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
    log_reg_classifier.fit(x_train, y_train)
    return log_reg_classifier

    # log_reg_classifier.fit(x_train, y_train)
def log_reg_coef(model):
    """Identify Coefficients"""
    print("===================================")
    print('Feature Coefficients')
    print("===================================")
    print(np.round(model.coef_,decimals=3))
    print("===================================")
    print('Feature Coefficients > 0')
    print("===================================")
    print(np.round(model.coef_,decimals=3) > 0)

def log_reg_cm(model, x_test, y_test):
    prediction = model.predict(x_test)

    # Calculate confusion_matrix
    cm = confusion_matrix(prediction, y_test)
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
