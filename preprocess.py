import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

prep = pd.DataFrame.from_csv('HR_comma_sep.csv', index_col=None)

# Is there are any missing values?
print("Preprocess 1:\n  ")
print("Check missing values result:\n", prep.isnull().any(), "\n ---------End---------\n")


# Renaming columns
class Rename:
    data_set = prep.rename(columns={'satisfaction_level': 'satisfaction',
                                    'last_evaluation': 'evaluation',
                                    'number_project': 'project',
                                    'average_montly_hours': 'hours',
                                    'time_spend_company': 'years',
                                    'Work_accident': 'accident',
                                    'promotion_last_5years': 'promotion',
                                    'sales': 'department',
                                    'left': 'left'
                                    })


# Move "left" to the first of the table
data_set = Rename.data_set
front = data_set['left']
data_set.drop(labels=['left'], axis=1, inplace=True)
data_set.insert(0, 'left', front)
data_set.head()

# Check data types
print("Preprocess 2: Check types \n")
print(data_set.dtypes, "\n ---------End---------\n")

# Use the label encoder to do the conversion (department & salary)
le_department = LabelEncoder()
le_salary = LabelEncoder()
le_department.fit(data_set.department)
le_salary.fit(data_set.salary)
data_set.department = le_department.transform(data_set.department)
data_set.salary = le_salary.transform(data_set.salary)

# Set y-axis and x-axis

y = data_set['left']
features = data_set[['satisfaction', 'hours', 'years', 'evaluation',
                     'project']]

# Normalize the transformed data set

# Use the label encoder to do the conversion
le_sales = LabelEncoder()
le_salary = LabelEncoder()
le_evaluation = LabelEncoder()

# normalization with min-max method
nor = MinMaxScaler()
features = nor.fit_transform(features)
print("Preprocess 3: Check normalized result \n")
print(features, "\n ---------End---------\n")

# Divide the data set into training set and testing set
x = features
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
print("Preprocess 4: Divide data into training set and testing set")
print('Training set volume:', x_train.shape[0])
print('Test set volume:', x_test.shape[0], "\n ---------End---------\n")
