from numpy import ravel
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn

# Use these lines if you want to include the enrolled students
#raw_data = fetch_ucirepo(id=697).data
#orig_X = raw_data.features
#orig_y = raw_data.targets

# Use these lines if you want to remove the enrolled students
raw_data = fetch_ucirepo(id=697).data.original
raw_data = raw_data.loc[raw_data['Target'] != 'Enrolled']
orig_y = raw_data['Target']
del raw_data['Target']
orig_X = raw_data

# choosing the input variables
input_names = ['Application order', 'Daytime/evening attendance', 'Previous qualification',
               'Previous qualification (grade)',
               'Admission grade', 'Displaced', 'Educational special needs','Debtor', 'Tuition fees up to date',
               'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
               'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
               'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
               'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
               'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
               'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
               'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
               'Unemployment rate', 'Inflation rate']

scales = {
    'Marital Status': 6,
    'Application mode': 57,
    'Application order': 9,
    'Course': 1000,
    'Daytime/evening attendance': 1,
    'Previous qualification': 43,
    'Previous qualification (grade)': 200,
    'Nacionality': 110,
    "Mother's qualification": 44,
    "Father's qualification": 44,
    "Mother's occupation": 195,
    "Father's occupation": 195,
    'Admission grade': 200,
    'Displaced': 1,
    'Educational special needs': 1,
    'Debtor': 1,
    'Tuition fees up to date': 1,
    'Gender': 1,
    'Scholarship holder': 1,
    'Age at enrollment': 15,
    'International': 1,
    'Curricular units 1st sem (credited)': 5,
    'Curricular units 1st sem (enrolled)': 5,
    'Curricular units 1st sem (evaluations)': 5,
    'Curricular units 1st sem (approved)': 5,
    'Curricular units 1st sem (grade)': 20,
    'Curricular units 1st sem (without evaluations)': 5,
    'Curricular units 2nd sem (credited)': 5,
    'Curricular units 2nd sem (enrolled)': 5,
    'Curricular units 2nd sem (evaluations)': 5,
    'Curricular units 2nd sem (approved)': 5,
    'Curricular units 2nd sem (grade)': 20,
    'Curricular units 2nd sem (without evaluations)': 5,
    'Unemployment rate': 100,
    'Inflation rate': 100,
    'GDP': 7
}

num_inputs = len(input_names)

X = orig_X[input_names]

# Data normalization, scale all data to the same level so that they have equal impact on the result
for name in X:
    X[name] /= scales[name]
y = orig_y

size = X.shape[0]

# separating the data into 4 folds plus another set of test values
folds = [X.iloc[:size//5].values, X.iloc[size//5:2*size//5].values, X.iloc[2*size//5:3*size//5].values, X.iloc[3*size//5:4*size//5].values]
folds_y = [ravel(y.iloc[:size//5].values), ravel(y.iloc[size//5:2*size//5].values), ravel(y.iloc[2*size//5:3*size//5].values), ravel(y.iloc[3*size//5:4*size//5].values)]

test_X = X.iloc[4*size//5:].values
test_y = ravel(y.iloc[4*size//5:].values)

neigh = knn(n_neighbors=9)#, weights='distance')

# Initializing some variables used for testing statistics
lr_stats = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0}
knn_stats = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0}
total_stats = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}
num_correct = 0

for validation_num in range(0, 4):
    # find the total size of all the training data to determine how big our arrays should be
    sample_size = 0
    for i in range(0, 4):
        if i != validation_num:
            sample_size += len(folds[i])

    X_cur = np.zeros(shape=(sample_size, folds[0].shape[1]))
    X_validation = folds[validation_num]
    y_cur = np.zeros(shape=(sample_size,), dtype=object)
    y_validation = folds_y[validation_num]
    cur = 0
    for i in range(0, 4):
        if i != validation_num:
            for j in range(0, len(folds[i])):
                # put all the values besides those in the ith fold into the X_cur array
                X_cur[cur] = folds[i][j]
                y_cur[cur] = folds_y[i][j]
                cur += 1

    # The actual knn function goes here
    neigh.fit(X_cur, y_cur)

    total_stats['All'] += len(folds[validation_num])
    for i in range(0, len(X_validation)):
        total_stats[y_validation[i]] += 1

        if neigh.predict(X_validation[i].reshape(1,-1)) == y_validation[i]:
            num_correct += 1
            knn_stats[y_validation[i]] += 1


print("accuracy: " + str(float(num_correct)/total_stats['All']))
print("accuracy for graduate: " + str(float(knn_stats['Graduate'])/total_stats['Graduate']))
#print("accuracy for enrolled: " + str(float(knn_stats['Enrolled'])/total_stats['Enrolled']))
print("accuracy for dropout: " + str(float(knn_stats['Dropout'])/total_stats['Dropout']))

#print("total grad: " + str(total_stats['Graduate']))
#print("total enrolled: " + str(total_stats['Enrolled']))
#print("total dropout: " + str(total_stats['Dropout']))

num_correct = 0
for i in range(0, len(test_X)):
    if neigh.predict(test_X[i].reshape(1,-1)) == test_y[i]:
        num_correct += 1

print("testing accuracy: " + str(float(num_correct)/len(test_X)))

