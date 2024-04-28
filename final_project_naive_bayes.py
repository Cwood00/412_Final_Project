from numpy import ravel
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB as nb

import sys
from datetime import datetime

numFolds = 5
include_enrolled = True

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

# choosing the input variables continuous variables are excluded for naive bayes
input_names = [
    # 'Marital Status',
    'Application order',
    # 'Application mode',
    # 'Course',
    'Daytime/evening attendance',
    # 'Previous qualification',
    # 'Previous qualification (grade)', Should not be used with naive bayes
    # 'Nacionality',
    # 'Mother\'s qualification',
    # 'Father\'s qualification',
    # 'Mother\'s occupation',
    # 'Father\'s occupation',
    # 'Admission grade', Should not be used with naive bayes
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    # 'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    # 'Curricular units 1st sem (grade)', Should not be used with naive bayes
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    # 'Curricular units 2nd sem (grade)', Should not be used with naive bayes
    'Curricular units 2nd sem (without evaluations)',
    # 'Unemployment rate', Should not be used with naive bayes
    # 'Inflation rate', Should not be used with naive bayes
    #'GDP' Should not be used with naive bayes
]

# Used to store result of each permutation
dtString = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"naive_bayes_{dtString}.txt"

outputFile = open(filename, 'a')

X = orig_X[input_names]
y = orig_y

size = X.shape[0]

# separating the data into training and testing data folds
training_X = X.iloc[:(numFolds - 1) * size // numFolds].values
training_y = ravel(y.iloc[:(numFolds - 1) * size // numFolds].values)

test_X = X.iloc[(numFolds - 1) * size // numFolds:].values
test_y = ravel(y.iloc[(numFolds - 1) * size // numFolds:].values)

# Initializing some variables used for testing statistics
correct_predictions = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}
total_predictions = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}
# nested maps outer map key is true class, inner map key is predicted class, inner map value is count
confusion_matrix = {'Graduate': {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0},
                    'Enrolled': {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0},
                    'Dropout': {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}}

start_time = datetime.now()
model = nb().fit(training_X, training_y)
end_time = datetime.now()
print(f"Took {(end_time - start_time).total_seconds()} seconds to train naive bayes, using {len(training_y)} samples", file=outputFile)

start_time = datetime.now()
predictions = model.predict(test_X)
end_time = datetime.now()
print(f"Took {(end_time - start_time).total_seconds()} seconds to test naive bayes, using {len(test_X)} samples", file=outputFile)

for i in range(len(predictions)):
    prediction = predictions[i]
    true_value = test_y[i]

    total_predictions[true_value] += 1
    total_predictions['All'] += 1
    confusion_matrix[true_value][prediction] += 1

    if prediction == true_value:
        correct_predictions[true_value] += 1
        correct_predictions['All'] += 1


print(f"Total testing accuracy = {(correct_predictions['All'] / total_predictions['All']) * 100}%", file=outputFile)
print(f"Graduate testing accuracy = {(correct_predictions['Graduate'] / total_predictions['Graduate']) * 100}%", file=outputFile)
if include_enrolled:
    print(f"Enrolled testing accuracy = {(correct_predictions['Enrolled'] / total_predictions['Enrolled']) * 100}%", file=outputFile)
print(f"Dropout testing accuracy = {(correct_predictions['Dropout'] / total_predictions['Dropout']) * 100}%", file=outputFile)
print(f"Out of {total_predictions['Graduate']} true graduate samples, "
      f"{confusion_matrix['Graduate']['Graduate']} were predicted graduate, "
      f"{confusion_matrix['Graduate']['Enrolled']} were predicted enrolled, and, "
      f"{confusion_matrix['Graduate']['Dropout']} where predicted dropout", file=outputFile)
if include_enrolled:
    print(f"Out of {total_predictions['Enrolled']} true Enrolled samples, "
          f"{confusion_matrix['Enrolled']['Graduate']} were predicted graduate, "
          f"{confusion_matrix['Enrolled']['Enrolled']} were predicted enrolled, and, "
          f"{confusion_matrix['Enrolled']['Dropout']} where predicted dropout", file=outputFile)
print(f"Out of {total_predictions['Dropout']} true Dropout samples, "
      f"{confusion_matrix['Dropout']['Graduate']} were predicted graduate, "
      f"{confusion_matrix['Dropout']['Enrolled']} were predicted enrolled, and, "
      f"{confusion_matrix['Dropout']['Dropout']} where predicted dropout", file=outputFile)

outputFile.close()  # Closing it to clear the buffer
