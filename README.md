# Music Information Retrieval (MIR) for Persian Music

## Introduction to MIR
**Music Information Retrieval (MIR)** is an interdisciplinary field that focuses on analyzing, organizing, and retrieving music-related data using computational methods. MIR techniques are widely used in applications such as genre classification, music recommendation, automatic transcription, and audio fingerprinting.

MIR leverages various machine learning and signal processing techniques to extract meaningful patterns from audio signals. Some common MIR tasks include:
- **Genre Classification**: Identifying the genre of a given music track.
- **Tempo and Rhythm Analysis**: Detecting beats and tempo.
- **Melody Extraction**: Recognizing the main melody of a song.
- **Audio Similarity and Recommendation**: Suggesting songs based on similarity metrics.

## Project Overview
This project applies MIR techniques to Persian music, particularly in genre classification. The primary goal is to develop machine learning models capable of accurately classifying Persian songs into five distinct genres:
- **Pop**
- **Rap**
- **Fusion**
- **Traditional (Classical Persian)**
- **Rock**

## Dataset
The dataset was collected by crawling Persian music websites, filtering and processing tracks to ensure quality. The dataset includes **3667 tracks**, distributed as follows:
- **Pop**: 640
- **Rap**: 168
- **Fusion**: 47
- **Traditional**: 201
- **Rock**: 63

Each track was processed by extracting a **30-second** audio segment from its midpoint to ensure consistency in analysis.

## Methodology
1. **Data Preprocessing**:
   - Audio files converted to **WAV format**.
   - Extracted **MFCC (Mel-Frequency Cepstral Coefficients)** features.
   - Normalized and structured data for model training.

2. **Machine Learning Models**:
   - **Neural Networks (NN)**
   - **Recurrent Neural Networks (RNN)**
   - **Support Vector Machines (SVM)**

3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, and F1-score were used to evaluate the models.
   - Model comparison was conducted to identify the best-performing approach.

## Results & Analysis
The trained models achieved the following classification accuracies:
- **Neural Network**: 85%
- **RNN**: 88%
- **SVM**: 81%

RNN performed best due to its ability to capture temporal dependencies in audio sequences.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas librosa scikit-learn tensorflow keras
```

## Future Work
- Expanding the dataset for better generalization.
- Exploring transformer-based models for improved accuracy.
- Implementing a web-based demo for real-time classification.

## Contributions
Contributions are welcome! Feel free to submit pull requests or report issues.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, reach out via GitHub issues or email: **shahrouzi_ali@yahoo.com**.

