import sys 
from PyQt5.QtWidgets import QApplication , QWidget , QPushButton , QLabel , QFileDialog , QVBoxLayout
import numpy as np 
import keras
import pydub
import os
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def export_wav (file_name):
    music_dir = os.path.abspath(file_name)
    music = pydub.AudioSegment.from_file(music_dir)
    duration = music.duration_seconds
    middel_of_segment = int (duration / 2)
    start_time = middel_of_segment - 15
    end_time =  middel_of_segment + 15
    segment = music[start_time * 1000 : end_time * 1000]
    segment = segment.export(f"E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\temp\\temp_Demo.wav",format = 'wav')
    return segment

def test_stats(file_name):

    y, sr = librosa.load(file_name)

    stats = {}
    has_entered_name = False

    features = {
        'spectral_centroids': librosa.feature.spectral_centroid(y=y, sr=sr)[0].ravel(),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].ravel(),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr)[0].ravel(),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr)[0].ravel(),
        'zcr': librosa.feature.zero_crossing_rate(y).ravel(),
        'flatness': librosa.feature.spectral_flatness(y=y),
        'rms': librosa.feature.rms(y=y)
    }

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_stft_{i+1}'] = chroma_stft[i]

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    features['beats'] = beats

    onset_env = librosa.onset.onset_strength(y=y, sr=sr).ravel()
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr).ravel()
    features['onset_env'] = onset_env
    features['onset_frames'] = onset_frames

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for ind in range(len(mfccs)):
        features[f'mfcc_{ind}'] = mfccs[ind]

    harmonic, percussive = librosa.effects.hpss(y)
    features['harmonic'] = harmonic
    features['percussive'] = percussive

    for key in features:
            if key == 'tempo':
                stats['tempo'] = features[key]
            elif 'chroma_stft' in key:
                stats[f'{key}_min'] = np.min(features[key])
                stats[f'{key}_mean'] = np.mean(features[key])
                stats[f'{key}_std'] = np.std(features[key])
            else:
                stats[f'{key}_max'] = np.max(features[key])
                stats[f'{key}_min'] = np.min(features[key])
                stats[f'{key}_mean'] = np.mean(features[key])
                stats[f'{key}_std'] = np.std(features[key])

    return stats

def create_csv(stats):
    stats.pop('onset_env_min')
    features = pd.DataFrame.from_dict([stats])
    features.to_csv('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\temp\\temp.csv')
    return features


class MusicClassifier(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Music Classifier")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.label = QLabel("Select a music file :")
        layout.addWidget(self.label)

        self.btn_browse = QPushButton("Select")
        self.btn_browse.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.btn_browse)

        self.label_result_nn = QLabel("Neural Network Model Result:")
        layout.addWidget(self.label_result_nn)

        self.label_result_rnn = QLabel("Recurrent Neural Network Model Result:")
        layout.addWidget(self.label_result_rnn)

        self.label_result_svm = QLabel("SVM Model Result:")
        layout.addWidget(self.label_result_svm)

        self.setLayout(layout)

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        filename, _ = file_dialog.getOpenFileName(self, 'Open Audio File', '', 'Audio Files (*.wav *.mp3)')

        if filename:
            file_path = export_wav(filename)
            stats = test_stats(file_path)
            features = create_csv(stats)
   
            if not features.empty:
                
                
                df = pd.read_excel('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Demo-new\\final_non_binary.xlsx')  

                X = df.drop(columns=['name', 'genre'])
                y = df['genre']


                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)


                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


                ovr_classifier = SVC(kernel='linear', decision_function_shape='ovr')
                ovr_classifier.fit(X_train, y_train)

                
                nn_model = keras.models.load_model('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Models\\NN_classifire_merge_V1.keras')
                rnn_model =  keras.models.load_model('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Models\\RNN_Classifire_merge_V1.keras')
                # svm_model = joblib.load('E:\\MyLesson\\Bachelor\\7th Term\\Introduction To Machine Learning\\Project\\Source\\Models\\Supervised_ovr.pkl')

                result_nn = nn_model.predict(features)
                # result_svm = svm_model.predict(features)
                result_rnn = rnn_model.predict(features)
                label_rnn = np.argmax(result_rnn)
                label_nn = np.argmax(result_nn)
                
                new_sample_scaled = scaler.transform(features)
                new_predictions = ovr_classifier.predict(new_sample_scaled)
                
                if label_rnn == 0 :
                    label_rnn ='Classic'
                elif label_rnn == 1 :
                    label_rnn = 'pop'
                elif label_rnn == 2 :
                    label_rnn ='rock'
                elif label_rnn == 3 :
                    label_rnn ='tal'   
                elif label_rnn == 4 :
                    label_rnn ='rap'
                if label_nn == 4 :
                    label_nn ='Classic'
                elif label_nn == 1 :
                    label_nn = 'pop'
                elif label_nn == 2 :
                    label_nn ='tal'
                elif label_nn == 3 :
                    label_nn ='rap'   
                elif label_nn == 0 :
                    label_nn ='rock'

            self.label_result_nn.setText(f"Neural Network Model Result: {label_nn}")
            self.label_result_rnn.setText(f"Recurrent Neural Network Model Result: {label_rnn}")
            self.label_result_svm.setText(f"SVM Model Result: {new_predictions}")
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MusicClassifier()
    window.show()
    sys.exit(app.exec_())
    