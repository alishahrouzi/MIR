import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

file_path = 'E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\final_non_binary.csv'



def extract_feature(file_path):
    try:
        features = list(pd.read_csv(file_path).columns[3:])
        return features
    except Exception as e :
        print(f'An error occurred while extracting features: {e}')
        return None
    

def load_data(dataset_path):
    try :
        data = pd.read_csv(dataset_path)
    except Exception as e :
        print(f'data not load ( {e} )')
        return None  , None
    features = extract_feature(file_path)
    if features is None :
        return None , None
    inputs = (np.array(data[features]))
    targets = (np.array(data['genre']))
    
    return inputs , targets


def prepare_datasets(test_size , validation_size):
    inputs , targets = load_data(file_path)
    
    inputs_train , inputs_test , targets_train , targets_test = train_test_split(inputs , targets ,test_size=test_size)
    inputs_train , inputs_validation , targets_train , targets_validation = train_test_split(inputs_train , targets_train , test_size=validation_size)

    inputs_train = inputs_train[... , np.newaxis]
    inputs_validation = inputs_validation[... , np.newaxis]
    inputs_test = inputs_test[... , np.newaxis]

    return inputs_train , inputs_validation , inputs_test , targets_train , targets_validation , targets_test




def Build_model(input_shape):
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(32 , (3,3) , activation='relu' , input_shape = input_shape , padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3) , strides=(2,2) , padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32 , (3,3) , activation='relu' , input_shape = input_shape , padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3) , strides=(2,2) , padding='same'))
    model.add(keras.layers.BatchNormalization()) 

    model.add(keras.layers.Conv2D(32 , (2,2) , activation='relu' , input_shape = input_shape , padding='same'))
    model.add(keras.layers.MaxPool2D((2, 2) , strides=(2,2) , padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64 , activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(5 , activation='softmax'))

    return model


if __name__ == '__main__' :
    inputs_train , inputs_validation , inputs_test , targets_train , targets_validation , targets_test = prepare_datasets(0.25 , 0.2)

    input_shape = (inputs_train.shape[0] , inputs_train.shape[1] , inputs_train.shape[2])
    model = Build_model(input_shape)

    optimizer = keras.optimizers.Adam(0.0001)
    model.compile(optimizer = optimizer ,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['Accuracy'])
    
    model.summary()
    

    model.fit(inputs_train , targets_train , validation_data = (inputs_validation , targets_validation) , batch_size = 32 , epochs = 50)
    
    test_error , test_accuracy = model.evaluate(inputs_test , targets_test , verbose = 1)
    print(f'accuracy on test : {test_accuracy}')
    