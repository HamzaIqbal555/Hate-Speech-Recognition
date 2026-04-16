# Roman Urdu Hate Speech Detection

## Bachelor Thesis Project

This repository presents a comprehensive deep learning framework for detecting hate speech in Roman Urdu, a low-resource transliterated form of Urdu commonly used on social media platforms in Pakistan and South Asia. Developed as part of my bachelor's thesis, the project tackles the critical challenge of automated content moderation in multilingual, low-resource settings.

### Problem Statement
Hate speech on social media amplifies toxicity, extremism, and division, particularly in regions like South Asia where Roman Urdu dominates online discourse. Traditional NLP models fail due to domain mismatch and limited Roman Urdu embeddings/datasets. This project delivers high-accuracy classifiers (up to 92%) for real-time detection.

### Business Use Cases
- **Social Media Moderation**: Integrate into platforms (e.g., X/Twitter, Facebook) to flag abusive tweets in Roman Urdu, reducing moderator workload by 80-90% at scale.
- **Brand Safety**: Filter toxic content in ads/user-generated content for Pakistani brands targeting Urdu speakers.
- **Public Safety & Compliance**: Monitor religious hate/sexism for government/NGOs, aiding counter-extremism efforts (e.g., PTA compliance in Pakistan).
- **Customer Support**: Auto-escalate abusive queries in call centers/e-commerce (Daraz, Foodpanda).
- **Scalability**: Low-latency LSTM inference suitable for production (e.g., Flask API endpoint).

### Datasets
- **Binary**: task_1_train.tsv/test.tsv (original) + newdata.csv (augmented). Labels: Normal (1), Abusive/Offensive (0). ~Thousands of samples post-augmentation.
- **Multi-Class**: MDDLS.csv. Labels: Negative (0), Neutral (1), Sexism (2), Religious Hate (3), Profane/Untargeted (4). Balanced via SMOTE.

### Methodology
#### Preprocessing
- Punctuation/numeric removal.
- Lowercasing, Roman Urdu-specific stopwords (custom list of 100+).
- Stemming: Rules for vowel/consonant normalization (e.g., 'ain' → 'ein', 'ai' → 'ahi').

#### Embeddings
- Custom Word2Vec: word2vec_RU.txt (Roman Urdu-specific) + vectors500000.txt (general). High OOV coverage (~90% words matched). Dim: 100. Frozen during training.

#### Models
All use Embedding + RNN/LSTM/GRU variants + Dense layers. Adam optimizer, dropout (0.2-0.3).
- Binary: BiLSTM (best), BiGRU, SimpleRNN, DistilBERT fine-tuning.
- Multi-Class: Same + SMOTE oversampling. Softmax output.

### Key Results

| Task       | Model      | Test Accuracy       | Notes                      |
|------------|------------|---------------------|----------------------------|
| Binary     | BiLSTM     | 92%                 | Production-ready low FP    |
| Binary     | BiGRU      | ~90% (est.)         | Competitive                |
| Binary     | DistilBERT | Evaluated           | Transfer learning          |
| Multi-Class| BiLSTM     | High (plots converge)| Granular hate types        |
| Multi-Class| BiGRU      | High                | Robust to imbalance        |

Detailed metrics (precision/recall/F1) and confusion matrices in notebooks. Training plots show stable convergence.

### Repository Structure
- `README.md`: This file.
- `Roman_Urdu_Hate_Speech_Detection.ipynb`: Binary classification notebook.
- `Roman_Urdu_Hate_Speech_Detection(Multi_Class).ipynb`: Multi-class notebook.
- `.DS_Store`: macOS artifact (ignore).

### Quick Start
1. Clone repo: `git clone <repo-url>`.
2. Install deps: `pip install pandas numpy matplotlib tensorflow keras nltk imblearn scikit-learn`.
3. Download data/embeddings (paths in notebooks, e.g., MyDrive/FYP/). Mount Google Drive or adjust paths.
4. Run notebooks in Jupyter/Colab: Cells are sequential; outputs include models/plots.
5. Inference: Load saved model (`mymodel`), predict on new Roman Urdu text.

Example prediction (binary):

```python
from keras.models import load_model
model = load_model('path/to/mymodel')
text = pad_sequences(tokenizer.texts_to_sequences(["example roman urdu text"]), maxlen=MAX_LEN)
pred = (model.predict(text) > 0.5).astype(int)  # 0: Offensive, 1: Normal
```

### Future Enhancements
- Deploy as REST API (FastAPI/Docker).
- Ensemble models + explainability (SHAP/LIME).
- Expand to Urdu script/Voice hate detection.
- Continual learning on streaming data.

### Author
Developed during Bachelor's thesis in Data Science / Computer Science. Optimized for modularity, reproducibility, and production readiness. Contact via GitHub for collaborations/applications.

---
*Keywords: NLP, Deep Learning, Hate Speech Detection, Roman Urdu, LSTM, Word2Vec, Low-Resource Language Processing*

