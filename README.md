# 📧 Comparison of Machine Learning Algorithms with Advanced Algorithms for Email Phishing Detection

## 📘 Overview
This project focuses on **detecting phishing emails** by comparing the performance of traditional **Machine Learning (ML) algorithms** and more **advanced models**.  
Each model is evaluated using two feature extraction techniques — **TF-IDF** and **Word2Vec embeddings** — to analyze which combination yields better classification results.

---

## 🧩 Included Models
The project includes four Python scripts, each representing a separate algorithm:

- **Logistic Regression** – `Logistic_Regression_Model.py`
- **Random Forest** – `Random_Forest_Model.py`
- **Decision Tree** – `Decision_Tree_Model.py`
- **MLP (Multilayer Perceptron Neural Network)** – `MLP_Model.py`

Each script:
- Loads and cleans the **Phishing_Email.csv** dataset  
- Performs **text preprocessing** (lowercasing, removing special characters, lemmatization, and stopword removal)  
- Trains two separate models:
  - One using **TF-IDF features**
  - One using **Word2Vec embeddings**
- Compares both using performance metrics and **3D bar graphs**

---

## 🧠 Objective
To evaluate and compare the efficiency of **classical ML algorithms** versus **advanced neural models** in identifying phishing emails.

---

## ⚙️ Requirements
Install all necessary libraries before running the code:
pip install pandas numpy matplotlib seaborn scikit-learn gensim nltk

You will also need to download required NLTK datasets:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

📊 Dataset
File Used: Phishing_Email.csv

Contains email text and corresponding labels (Email Text, Email Type)

Email Type indicates whether an email is phishing or legitimate

Ensure the dataset is in the same directory as your Python scripts.

🧮 Workflow Summary
Data Preprocessing

Text cleaning (lowercasing, punctuation removal)

Lemmatization and stopword removal

Tokenization for Word2Vec

Feature Extraction

TF-IDF Vectorizer – Converts text into numerical feature vectors

Word2Vec Embeddings – Captures semantic relationships between words

Model Training

Classical ML: Logistic Regression, Random Forest, Decision Tree

Advanced ML: MLP Neural Network

Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-Score

Visualization: 3D Bar Graphs for metric comparison

Confusion Matrices and Classification Reports

📈 Outputs
Performance Metrics displayed for TF-IDF and Word2Vec models

3D visualizations comparing accuracy, precision, recall, and F1-score

Confusion matrices for visualizing true vs predicted classifications

Text-based classification reports for each model

🧾 Results Summary
Algorithm	Feature Type	Accuracy	Key Insights
Logistic Regression	TF-IDF / Word2Vec	Moderate	Simple and interpretable; performs well with TF-IDF
Random Forest	TF-IDF / Word2Vec	High	Handles non-linearity; good feature importance
Decision Tree	TF-IDF / Word2Vec	Moderate	Easy visualization; may overfit
MLP (Neural Network)	TF-IDF / Word2Vec	High	Learns complex relationships; better generalization

🚀 Future Enhancements
Integrate deep learning models (e.g., LSTM, BERT)

Optimize hyperparameters for improved performance

Add real-time phishing detection web interface

Include cross-validation for model reliability

👩‍💻 Author
Bhuvaneshwari C
B.Tech – Computer Science and Engineering

📝 License
This project is open-source and available under the MIT License.

⭐ If you find this project useful, please give it a star on GitHub!
