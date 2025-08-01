# Disaster Tweets Classification (DL + Word Embeddings)
This project focuses on binary classification of tweets to determine whether they refer to real disasters.
It uses traditional ML (SVM with TF-IDF and Word2Vec) and Deep Learning (Conv1D + Embedding) techniques.

## 🗂️ Dataset 
- Source: [Kaggle - Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- Binary Labels:
  - `1` - disaster-related tweet
  - `0` - not disaster-related

## 📚 Technologies % Libraries
- Python 3.x
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- Tensorflow / Keras
- spacy with `en_core_web_lg`
- GloVe / spacy word embeddings
- Custom library: [`preprocess_tr`](https://github.com/troyhunterz/preprocess_tr)
- textblob, wordcloud

## 🏷️ Features
- Data visualization (distribution, word clouds, char/word stats)
- Preprocessing with regular expression, HTML removal, normalization
- Feature extraction:
  - TF-IDF
  - Word2Vec
  - Deep Learning Embedding + Conv1D

- Classifiers:
  - LinearSVC (on TF-IDF & Word2Vec)
  - Conv1D-based Keras model

## 🤖 Model Training
### 1. TF-IDF + SVM
- Vectorization using `TfidfVectorizer`
- Classification via `LinearSVC`

### 2. Word2Vec + SVM
- Sentence vectors via spacy (`en_core_web_lg`)
- Flattened to 300-dim vectors
- Classification with `LinearSVC`

### 3. Deep Learning + Embedding
- Tokenization and padding
- Embedding layer
- Conv1D → MaxPooling1D → Dropout
- Dense (ReLu) → Dropout → Dense (ReLu)
- GlobalMaxPooling1D
- Output Dense (sigmoid)

## 🔍 Evaluation
Used metrics:
- Accuracy
- F1-score
- Confusion Matrix
- Precision / Recall

## 🧪 Sample Prediction
```python
get_results("There is a fire in the forest")      # → [disaster]
get_results("Happy birthday!")                    # → [not disaster]
get_results("Vaccination campaign is starting")   # → [disaster]
```

## ⚙️ Setup Instructions
```bash
git clone https://github.com/troyhunterz/disaster-tweets-dl.git
cd disaster-tweets-dl
```

## 🧾 License
This project is licensed under the MIT License.

## 👤 Author
troyhunterz
Email: ann0nfolder@gmail.com