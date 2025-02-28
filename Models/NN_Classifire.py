#merge without Duplicate file : 1 layer hidden (256 , relu) , best accuracy : about 60 percent

import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import to_categorical
import math 
import json 
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt 
import csv
import pandas as pd
import openpyxl as oxl


# file_path_1 = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\tal-V1.csv')
# file_path_2 = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\rap-V1.csv')
# file_path_3 = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\rock-V1.csv')
# file_path_4 = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\classic-V1.csv')
# file_path_5 = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\pop-V1-B.csv')
# file_path = pd.concat(map(pd.read_csv, [file_path_1 , file_path_2 , file_path_3 , file_path_4 , file_path_5]),ignore_index = False)
# file_path.to_csv('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\merge-v2.csv')


file_path = os.path.abspath('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\merge-v2.csv')


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
    axs[0].plot(history.history['accuracy'] , label = 'train accuracy')
    axs[0].plot(history.history['val_accuracy'] , label = 'test accuracy')
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

def make_json_from_csv (file_path , json_file_path):
        try :
            data = pd.read_csv(file_path)
            json_data = data.to_json()
            with open ('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\json_classic.json' , 'w') as jsonf:
                jsonf.write(json.dumps(json_data))
            print('json file writted')
        except Exception as e :
            print(f'an erorr occurred : {e}')  


def make_json_from_excel(file_path , json_file_path):
    try :
        data = pd.read_excel(file_path)
        json_data = data.to_json(orient='records' , indent= 4)
        with open(json_file_path , 'r' ,) as jsonf:
            jsonf.write(json.dumps(json_data))
        print('json file writted')
    except Exception as e :
        print(f'an erorr occurred : {e}')            


if __name__ == '__main__':
    inputs , targets = load_data(file_path)
    input_train , input_test , target_train , target_test = train_test_split(inputs , targets , test_size = 0.3)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (164 , )),
        
        keras.layers.Dense(512 , activation = 'relu' , kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256 , activation = 'relu' , kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256 , activation = 'relu' , kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256 , activation = 'relu' , kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64 , activation = 'relu' , kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(6 , activation= 'softmax')
    ])
    optimizers = keras.optimizers.Adam(learning_rate=0.0001)
    loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(
                  optimizer = optimizers,
                  loss = loss ,
                  metrics = ['accuracy'])
    model.summary()
    

    history = model.fit(input_train , target_train , 
                        validation_data = (input_test , target_test),
                        epochs = 50 ,
                        batch_size = 32)
    
    plot_history(history)
    
    
    # model.save('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Models\\NN_classifire_merge_V2.keras')