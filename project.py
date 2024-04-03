import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# [X] Marital status
# [ ] Application mode
# [X] Application order
# [ ] Course
# [X] Daytime/evening attendance
# [ ] Previous qualification
# [ ] Previous qualification (grade)
# [X] Nacionality
# [?] Mother's qualification
# [?] Father's qualification
# [ ] Mother's occupation
# [ ] Father's occupation
# [X] Admission grade
# [X] Displaced
# [X] Educational special needs
# [X] Debtor
# [X] Tuition fees up to date
# [X] Gender
# [X] Scholarship holder
# [X] Age at enrollment
# [X] International
# [X] Curricular units 1st sem (credited)
# [X] Curricular units 1st sem (enrolled)
# [ ] Curricular units 1st sem (evaluations)
# [ ] Curricular units 1st sem (approved)
# [ ] Curricular units 1st sem (grade)
# [ ] Curricular units 1st sem (without evaluations)
# [ ] Curricular units 2nd sem (credited)
# [ ] Curricular units 2nd sem (enrolled)
# [ ] Curricular units 2nd sem (evaluations)
# [ ] Curricular units 2nd sem (approved)
# [ ] Curricular units 2nd sem (grade)
# [ ] Curricular units 2nd sem (without evaluations)
# [ ] Unemployment rate
# [ ] Inflation rate
# [ ] GDP
# [ ] Target

columns = [
    # "Marital Status",
    # "Application order",
    # "Daytime/evening attendance",
    # "Nacionality",
    "Admission grade",
    # "Displaced",
    # "Educational special needs",
    # "Debtor",
    # "Tuition fees up to date",
    # "Gender",
    # "Scholarship holder",
    # "Age at enrollment",
    # "International",
    # "Curricular units 1st sem (credited)",
    # "Curricular units 1st sem (enrolled)"
]

mapping = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2,
}

# data (as pandas dataframes)
X = pd.DataFrame(predict_students_dropout_and_academic_success.data.features, columns=columns)
y = pd.DataFrame(predict_students_dropout_and_academic_success.data.targets, columns=["Target"])

y = np.vectorize(mapping.get)(y)

from matplotlib import pyplot as plt

y = np.ravel(y)
selected_feature = X.columns[0]
X_selected = X[selected_feature].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.scatter(X_train, y_train)
plt.show()