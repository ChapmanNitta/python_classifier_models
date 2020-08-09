import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import log_reg as lr
import knn
import decision_tree as dt

# Display the max amount of columsn
pd.set_option('display.max_columns', None)

# Point to working directory and read telco churn csv
churn = 'Telco Churn'
input_dir = os.path.dirname(__file__)
churn_file = os.path.join(input_dir, churn + '.csv')

# Load churn data to dataframe and remove the customer ID value
data = pd.read_csv(churn_file)
data = data.iloc[:,1:]

dummied_data = pd.get_dummies(data, columns = ['gender', 'Partner', 'Dependents','PhoneService','MultipleLines','InternetService',
'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn'], drop_first = True)

dummied_data['TotalCharges'] = pd.to_numeric(dummied_data.TotalCharges, errors = 'coerce')
dummied_data.drop(['TotalCharges'], axis = 1, inplace = True)

features = dummied_data.iloc[:,:-1].columns.values
x = dummied_data[features].values
y = dummied_data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=0)

#==============================================================================
# Logistic Regression Using all features
#==============================================================================
print('Logistic Regression Using All Features')
log_reg_classifier = lr.log_reg(x_train, y_train)
# Compute Accuracy Scores
lr.log_reg_cm(log_reg_classifier, x_test, y_test)
# Compute Coefficients
print("                                   ")
print("                                   ")
print("===================================")
print('Logistic Regression using features with the highest Weights')
print(features)
lr.log_reg_coef(log_reg_classifier)
#==============================================================================
# Logistic Regression Using Highest Features by Weights
#==============================================================================
new_features = ['SeniorCitizen', 'MonthlyCharges', 'MultipleLines_No phone service',
'MultipleLines_Yes', 'InternetService_Fiber optic',
'PaperlessBilling_Yes', 'PaymentMethod_Electronic check',
'PaymentMethod_Mailed check']

x2 = dummied_data[new_features].values
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.50, random_state=0)

# Logistic Regression
log_reg_classifier_limited_features = lr.log_reg(x2_train, y2_train)
# Compute Accuracy Scores
lr.log_reg_cm(log_reg_classifier_limited_features, x2_test, y2_test)

##==============================================================================
## KNN Using all features
##==============================================================================
print("                                   ")
print("                                   ")
print("===================================")
print('KNN Using All Features')
print("===================================")
print('Accuracy Rates & Plot')
print("===================================")

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

knn_x_train, knn_x_test, knn_y_train, knn_y_test = train_test_split(x, y, test_size=0.50, random_state=0)

knn.plot_accuracy(knn_x_train, knn_x_test, knn_y_train, knn_y_test)
print('''While 20 neighbors had a marginal increase in accuracy, it didn't
     outweigh the performance hit that would be taken at 15 neighbors''')
print("===================================")
print('KNN CM Using All Features & k=15')
print("===================================")

knn.knn_cm(knn_x_train, knn_x_test, knn_y_train, knn_y_test, 15)
#==============================================================================
# KNN using the highest weighted features
#==============================================================================

scaler.fit(x2)
x2 = scaler.transform(x2)

knn_x2_train, knn_x2_test, knn_y2_train, knn_y2_test = train_test_split(x2, y, test_size=0.50, random_state=0)

print("                                   ")
print("                                   ")
print("===================================")
print('''KNN Using SeniorCitizen, MonthlyCharges, MultipleLines_No phone service, MultipleLines_Yes, InternetService_Fiber optic, PaperlessBilling_Yes, PaymentMethod_Electronic check,PaymentMethod_Mailed check
''')
print("===================================")
print('Accuracy Rates & Plot')
print("===================================")
knn.plot_accuracy(knn_x2_train, knn_x2_test, knn_y2_train, knn_y2_test)
print("===================================")
print('''KNN Using SeniorCitizen, MonthlyCharges, MultipleLines_No phone service, MultipleLines_Yes, InternetService_Fiber optic, PaperlessBilling_Yes, PaymentMethod_Electronic check,PaymentMethod_Mailed check
 k=19''')
print("===================================")
knn.knn_cm(knn_x2_train, knn_x2_test, knn_y2_train, knn_y2_test, 19)

#==============================================================================
# Decision Tree
#==============================================================================
print("                                   ")
print("                                   ")
print("===================================")
print('Decision Tree Using All Features')
print("===================================")
x_train_dt, x_test_dt , y_train_dt, y_test_dt = train_test_split(x,y, test_size=0.5, random_state=0)

dt.decision_tree(x_train_dt, x_test_dt , y_train_dt, y_test_dt)

print("                                   ")
print("                                   ")
print("===================================")
print('Decision Tree Using Highest Weighted Features')
print("===================================")

dt.decision_tree(x2_train, x2_test, y2_train, y2_test)