import pandas as pd 
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import numpy as np
import os
import librosa as lib
import audioread as audio


# analayze_feature_second_edition = pd.read_csv('D:\\MyLesson\\کارشناسی\\ترم هفت\\مبانی یادگیری ماشین\\Project\\Source\\CSV\\persian_music_file_data_second_edition.csv')
analayze_feature_merge_edition = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\Merged.csv')
analayze_feature_sample_edition = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Sample\\clusters_sample_5.csv')
sample = pd.read_csv('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\CSV\\features_30_sec.csv')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14 
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#000000'

def create_histogram_data (property):
    func_histogram = px.histogram(analayze_feature_merge_edition,
                 x = f'{property}',
                 marginal = 'box',
                 nbins = 47,
                 title = 'Distribution of '+ property)
    func_histogram.update_layout(bargap = 0.6)
    func_histogram.show()
    
def create_scatter_data(property):
    func_scatter = px.scatter(analayze_feature_merge_edition,
                              x = f'{property}',
                              y = 'name',
                              opacity = 1,
                              title = f'Distribution of {property}')
    func_scatter.update_layout(bargap = 0.6)
    func_scatter.show()    


#Itrate:
# for i in np.arange(0,int(len(analayze_feature_sample_edition.columns)),18):
#     create_histogram_data(analayze_feature_sample_edition.columns[i+3])
#     create_scatter_data(analayze_feature_sample_edition.columns[i+3])



#Feature_Correlation:

list_corr = []
list_property1 = []
list_all = [analayze_feature_merge_edition.columns.drop(['no','name','genre'])]
list_non_used = list(i for i in list_all if i not in list_property1)
def corr_feature() :
    for i in np.arange(int(len(analayze_feature_merge_edition.columns))):
            if i + 3 < len(analayze_feature_merge_edition.columns) :    
                property1 = analayze_feature_merge_edition.columns[i+3]
                for j in np.arange(len(analayze_feature_merge_edition.columns)):
                    if j + 3 < len(analayze_feature_merge_edition.columns) :
                        property2 = analayze_feature_merge_edition.columns[j+3]
                        corr = analayze_feature_merge_edition[property1].corr(analayze_feature_merge_edition[property2])
                        if property1 is not property2 and corr > 0.87 or corr < -0.87 :
                            print(f'Correlation {property1} and {property2} = ' , corr)
                            if not property1 in list_property1:
                                list_property1.append(property1)
                            if not property2 in list_property1 :
                                list_corr.append(corr)

corr_feature()

array_zeros = list(0 for i in range(len(analayze_feature_merge_edition.columns)))
for i in list_property1:
    array_zeros[list(analayze_feature_merge_edition.columns).index(i)] = 1
    
print(list_non_used)
print('List Features : ' ,list_property1,len(list_property1))
print(array_zeros)

#Generate_Spectogarm:

input_dir = ('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo')
output_dir = ('D:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Analytics\\Picture_Spectogram')  

#Spectogram:

def crc_sepec():
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_dir, filename)
            y , sr = lib.load(file_path)
            D = lib.stft(y)
            S_db = lib.amplitude_to_db(np.abs(D), ref = np.max)
            plt.figure(figsize=(10,5),facecolor='white')
            lib.display.specshow(S_db)
            plt.colorbar(format = '%+2.0f dB')
            plt.title(f'{file_path} Spectogram')
            plt.tight_layout()
            plt.gca().set_facecolor('white')
            output_file_path = os.path.join(output_dir,filename[:-4] + '_Spectogram.png')
            plt.savefig(output_file_path , dpi = 300)
            plt.close()


#melSpectogram:

def crc_melspec():
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_dir, filename)
            y , sr = lib.load(file_path)
            mels = lib.feature.melspectrogram(y=y , sr=sr)
            plt.figure(figsize=(10,5), facecolor='white')
            lib.display.specshow(mels , cmap = 'viridis')
            plt.colorbar(format = '%+2.0f dB')
            plt.title(f'{file_path} Spectogram')
            plt.tight_layout()
            plt.gca().set_facecolor('white')
            output_file_path = os.path.join(output_dir,filename[:-4] + '_melSpectogram.png')
            plt.savefig(output_file_path , dpi = 300)
            plt.close()

# crc_sepec()
# crc_melspec()

# var = []
# fild = []
# for i in range(3,163):
#     DataFrame.drop(analayze_feature_merge_edition).append(fild)
#     np.var(fild).append(var)
# if var >= 1 :
#     print(var)