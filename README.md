# Email Spam Detection using NLP
This project helps detect whether an email or SMS message is spam or not, using natural language processing and machine learning. It is designed as a clear, beginner-friendly mini-project for students to learn how NLP can be applied to real-world problems.

## What this project does
- Classifies messages as spam or not spam based on their content
- Cleans and prepares text data using text preprocessing
- Converts text into numerical data using Bag-of-Words
- Trains a Naive Bayes classifier on real-world SMS data
- Allows live testing of custom email from your terminal

## Tech Stack
Language: Python
Libraries: pandas, scikit-learn, nltk
Concepts: Natural Language Processing (NLP), Bag-of-Words, Naive Bayes Classification, Text Preprocessing

## Getting Started

1. Install dependencies

Make sure Python is installed on your system, then run:

pip install -r requirements.txt

2. Prepare the dataset

Download the SMS Spam Collection dataset (spam.csv) from Kaggle and place it in your project folder.

3. Running the project

Using Python Script:

Run:

python spam_detection.py

The model will train and display accuracy and a classification report. You can type your own email or SMS message to check if it is spam. Type 'exit' to quit the program.

Using Jupyter Notebook:

Run:

jupyter notebook

Open spam_detection.ipynb to see each step, modify, and experiment for a deeper understanding.

## Example Usage

Enter an email to check if it is spam (or type 'exit' to quit):
Congratulations! You have won a free ticket to Bahamas. Claim now!

Output:
Spam

