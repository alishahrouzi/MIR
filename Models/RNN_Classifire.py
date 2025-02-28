import numpy as np 
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import os
import pandas as pd

file_path = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\final_non_binary.csv')

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

def plot_history(history):
    fig , axs = plt.subplots(2)
    axs[0].plot(history.history['Accuracy'] , label = 'train accuracy')
    axs[0].plot(history.history['val_Accuracy'] , label = 'test accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].legend(loc = 'lower right')
    axs[0].set_title('accuracy eval')
    
    axs[1].plot(history.history['loss'] , label = 'train erorr')
    axs[1].plot(history.history['val_loss'] , label = 'test erorr')
    axs[1].set_ylabel('erorr')
    axs[1].set_xlabel('epoch')
    axs[1].legend(loc = 'upper right')
    axs[1].set_title('\nerorr eval')

    plt.show()

def prepare_datasets(test_size , validation_size):
    X , y = load_data(file_path)
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=test_size)
    X_train , X_validation , y_train , y_validation = train_test_split(X_train , y_train , test_size=validation_size)
    
    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    
    return X_train , X_validation , X_test , y_train , y_validation , y_test

def Build_model(input_shape):
    model = keras.Sequential()
    
    model.add(keras.layers.LSTM(64 , input_shape = input_shape , return_sequences=True))
    model.add(keras.layers.LSTM(64))
    
    model.add(keras.layers.Dense(64 , activation= 'relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(5 , activation='softmax'))
    
    return model



if __name__ == '__main__' :
    X_train , X_validation , X_test , y_train , y_validation , y_test = prepare_datasets(0.25 ,0.2)
    
    input_shape = ((X_train.shape[1] , X_train.shape[2]))
    model = Build_model(input_shape)
    
    optimizer = keras.optimizers.Adam(0.0001)
    model.compile(optimizer = optimizer , loss = 'sparse_categorical_crossentropy' , metrics = ['Accuracy'])
    
    model.summary()
    
    history= model.fit(X_train ,y_train , validation_data = (X_validation , y_validation) , batch_size = 32 , epochs = 50)
    
    plot_history(history)
    
    test_loss , test_acc = model.evaluate(X_test , y_test , verbose = 2)
    print('\nTest Accuracy: ' , test_acc)
    

    
    model.save('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Models\\RNN_Classifire_merge_V1.keras')
    # print(np.argmax(model.predict(X_test)[0]))