import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.wordpunct_tokenize(text)
    text = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.markdown("<h1 style='font-size: 40px; color: white;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Larger input area
input_sms = st.text_area("Enter the message:", height=150)

# Predict button
if st.button("Predict", use_container_width=True):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.markdown("<h3 style='color: red;'>🚫 This message is <strong>SPAM</strong></h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>✅ This message is <strong>NOT SPAM</strong></h3>", unsafe_allow_html=True)
