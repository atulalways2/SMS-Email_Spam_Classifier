# SMS-Email_Spam_Classifier
Spam Classifier
This project is a machine learning-based SMS spam classifier that identifies messages as either "spam" or "ham" (not spam). The project includes a Jupyter Notebook detailing the model development process and a simple web application for user-friendly classification.

Features
Data Preprocessing: Cleans and transforms raw SMS text data for machine learning.
Model Training: Trains a Multinomial Naive Bayes classifier on the preprocessed data.

Model Evaluation: Assesses model performance using metrics like accuracy, precision, and a confusion matrix.

Interactive Web App: A Streamlit application that allows users to input a message and get an instant spam/ham classification.

Technologies Used
Programming Language: Python

Libraries:
pandas: For data manipulation and analysis.
numpy: For numerical operations.
nltk: For text preprocessing and tokenization.
scikit-learn: For building and evaluating the machine learning model.
streamlit: For creating the interactive web application.
matplotlib.pyplot and seaborn: For data visualization.

File Structure
SpamDetection1.ipynb: A Jupyter Notebook containing the entire process of data cleaning, preprocessing, model training, and evaluation.

spam.csv: The dataset used for training and testing the model.

model.pkl: The serialized machine learning model.

vectorizer.pkl: The serialized CountVectorizer object used for text vectorization.

SpamClassifierApp.py: The Python script for the Streamlit web application.

Data
The dataset spam.csv consists of 5572 SMS messages and their labels. The two main columns used are v1 (renamed to Target) for the labels ("ham" or "spam") and v2 (renamed to Text) for the message content.

Methodology
The project follows a standard machine learning workflow for text classification:

Data Loading and Cleaning: The spam.csv file is loaded into a pandas DataFrame. Irrelevant columns are dropped, and duplicate messages are removed.

Text Preprocessing: The raw text data is transformed by converting it to lowercase, tokenizing it into individual words, and filtering out non-alphanumeric characters.

Vectorization: The preprocessed text is converted into a numerical format using CountVectorizer. This creates a matrix where each message is represented by the frequency of its words.

Model Training: A Multinomial Naive Bayes model is trained on the vectorized data.

Model Deployment: The trained model and vectorizer are saved as .pkl files and loaded into a web application built with Streamlit.
