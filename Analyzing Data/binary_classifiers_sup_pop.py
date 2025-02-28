from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import os
import sys

def data_set_maker():
    os.chdir('csvs')

    initial_dataset = pd.read_csv('pop.csv')
    initial_dataset['genre'] = 'pop'
    temp_lst = os.listdir()

    temp_lst.remove('pop.csv')
    temp_lst.remove('Merged.xlsx')
    temp_lst.remove('New folder')
    new_dataset = initial_dataset
    for csv in temp_lst:
        
        if csv.endswith('.xlsx'):
            secondary_set = pd.read_excel(csv)
        else:
            secondary_set = pd.read_csv(csv)
            
        secondary_set['genre'] = 'not_pop'
        new_dataset = pd.concat([new_dataset,secondary_set],ignore_index=True)


    new_dataset.to_csv('pop.csv', index=False)
    os.chdir('..')

data_set_maker()


os.chdir('csvs')

df = pd.read_csv('pop.csv')


# Step 2: Perform cross-validation split
X = df.drop(columns=['name','no','genre','onset_env_min'])
y = df['genre']


# Step 2.1: Normalize or standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train and test a classifier
pop_classifier =  SVC(kernel='linear')
pop_classifier.fit(X_train, y_train)

predictions = pop_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Step 4: Load the new dataset


new_dataset = pd.read_excel('Merged.xlsx').drop(columns=['no','genre'])
dropped = new_dataset.pop('name')

# Step 5: Normalize or standardize the features of the new dataset
new_X_scaled = scaler.transform(new_dataset)

# Step 6: Binary classification of the new dataset
new_predictions = pop_classifier.predict(new_X_scaled)

# Step 7: Add class labels to the new dataset
new_dataset['predicted_class'] = new_predictions
new_dataset = pd.concat([dropped,new_dataset], axis=1)


# Step 8: Save the modified dataset to a new CSV file
new_dataset.to_csv('predictions_pop.csv', index=False)