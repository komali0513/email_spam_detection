import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text_clean'] = df['text'].apply(preprocess)

cv = CountVectorizer()
X = cv.fit_transform(df['text_clean'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

while True:
    text = input("\nEnter an email to check if it is spam (or type 'exit' to quit):\n")
    if text.lower() == 'exit':
        break
    text_clean = preprocess(text)
    vect = cv.transform([text_clean])
    prediction = model.predict(vect)[0]
    print("Spam" if prediction == 1 else "Not Spam")
