from numpy import ravel
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from datetime import datetime

include_enrolled = False

if include_enrolled:
    raw_data = fetch_ucirepo(id=697).data
    orig_X = raw_data.features
    orig_y = raw_data.targets
else:
    raw_data = fetch_ucirepo(id=697).data.original
    raw_data = raw_data.loc[raw_data['Target'] != 'Enrolled']
    orig_y = raw_data['Target']
    del raw_data['Target']
    orig_X = raw_data

# choosing the input variables
input_names = [
    'Application order',
    'Daytime/evening attendance',
    # 'Previous qualification',
    'Previous qualification (grade)',
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate'
]

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

training_X = X.iloc[:4 * size // 5].values
training_y = ravel(y.iloc[:4 * size // 5].values)

test_X = X.iloc[4*size//5:].values
test_y = ravel(y.iloc[4*size//5:].values)

# Used to store result of each permutation
dtString = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{dtString}.txt"

outputFile = open(filename, 'a')

k_values = (1, 3, 5, 7, 9, 11, 13, 15)
average_prediction_accuracies = []

# learn optimal k, using cross validation
start_time = datetime.now()
for k in k_values:
    neigh = knn(n_neighbors=k)#, weights='distance')

    # Initializing some variables used for testing statistics
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

    average_prediction_accuracies.append(float(num_correct)/total_stats['All'])

    print(f"Validation on k = {k}", file=outputFile)
    print("accuracy: " + str(float(num_correct)/total_stats['All']), file=outputFile)
    print("accuracy for graduate: " + str(float(knn_stats['Graduate'])/total_stats['Graduate']), file=outputFile)
    if include_enrolled:
        print("accuracy for enrolled: " + str(float(knn_stats['Enrolled'])/total_stats['Enrolled']), file=outputFile)
    print("accuracy for dropout: " + str(float(knn_stats['Dropout'])/total_stats['Dropout']), file=outputFile)

end_time = datetime.now()
print(f"Took {(end_time - start_time).total_seconds()} seconds to perform cross validation for {len(k_values)} k values", file=outputFile)

#print("total grad: " + str(total_stats['Graduate']))
#print("total enrolled: " + str(total_stats['Enrolled']))
#print("total dropout: " + str(total_stats['Dropout']))

optimal_k = k_values[np.argmax(average_prediction_accuracies)]
neigh = knn(n_neighbors=optimal_k)#, weights='distance')
neigh.fit(training_X, training_y)

print(f"Optimal k value = {optimal_k}" , file=outputFile)

start_time = datetime.now()
predictions = neigh.predict(test_X)
end_time = datetime.now()
print(f"Took {(end_time - start_time).total_seconds()} seconds to test knn, using {len(test_X)} samples", file=outputFile)

correct_predictions = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}
total_predictions = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}

for i in range(len(predictions)):
    prediction = predictions[i]
    true_value = test_y[i]

    total_predictions[true_value] += 1
    total_predictions['All'] += 1

    if prediction == true_value:
        correct_predictions[true_value] += 1
        correct_predictions['All'] += 1

print(f"Total testing accuracy = {(correct_predictions['All'] / total_predictions['All']) * 100}%", file=outputFile)
print(f"Graduate testing accuracy = {(correct_predictions['Graduate'] / total_predictions['Graduate']) * 100}%", file=outputFile)
if include_enrolled:
    print(f"Enrolled testing accuracy = {(correct_predictions['Enrolled'] / total_predictions['Enrolled']) * 100}%", file=outputFile)
print(f"Dropout testing accuracy = {(correct_predictions['Dropout'] / total_predictions['Dropout']) * 100}%", file=outputFile)

