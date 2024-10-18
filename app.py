from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Ensure nltk stopwords are downloaded
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model/logistic_regression_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Preprocess the text (from your earlier code)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        cleaned_text = preprocess_text(news_text)
        X_new = vectorizer.transform([cleaned_text])
        prediction = model.predict(X_new)
        label = 'True' if prediction[0] == 1 else 'False'
        return render_template('index.html', prediction=label, news_text=news_text)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
