import nltk
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier  # Explicitly import VotingClassifier
from nltk.corpus import stopwords  # Import stopwords corpus
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer
import pickle


def download_nltk_resources():
    """Downloads necessary NLTK resources if not already available."""

    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        print("Error downloading NLTK resources:", e)


def clean_text(text):
    
        """Preprocesses text using NLTK and other techniques.
        
        Args:
        text (str): The text to be preprocessed.
        
        Returns:
        str: The preprocessed text.
        """
        
        download_nltk_resources()  # Download NLTK resources if needed
        
        text = text.lower()  # Lowercase
        text = re.sub('\[.*?\]', '', text)  # Remove text in square brackets
        text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove links
        text = re.sub('<.*?>+', '', text)  # Remove HTML tags
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
        text = re.sub('\n', '', text)  # Remove newlines
        text = re.sub(r'\d+', '', text)  # Remove numbers
        
        # Tokenize, remove stopwords, and lemmatize
        stop_words = stopwords.words('english')
        tokens = nltk.word_tokenize(text)
        filtered_words = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        cleaned_text = [lemmatizer.lemmatize(word) for word in filtered_words]
        
        return ' '.join(cleaned_text)



def load_or_train_model(model_path):
    """Loads a trained model from a pickle file if it exists,
       otherwise trains a new model and saves it."""

    try:
        with open(model_path, 'rb') as f:
            cv, classifier = pickle.load(f)
        print("Loaded model from", model_path)
    except FileNotFoundError:
        print("Model not found. Training a new model...")

        # Load and preprocess data 
        try:
            data = pd.read_csv("newdataset.csv")
            data['processed_content'] = data['Original Content'].apply(lambda x: clean_text(x))

            mapping = {'depression': 0, 'positive': 1, 'anger': 2}
            data['emotion'] = data['Emotion'].map(mapping)

            x = data['processed_content']
            y = data['emotion']
            data.drop(['Emotion'], axis=1, inplace=True)
            data.drop(['Original Content'], axis=1, inplace=True)

            cv = CountVectorizer()
            x = cv.fit_transform(x)

            # Train the ensemble model
            models = []
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

            from sklearn.naive_bayes import MultinomialNB
            NB = MultinomialNB()
            NB.fit(X_train, y_train)
            models.append(('naive_bayes', NB))

            from sklearn import svm
            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            models.append(('svm', lin_clf))

            from sklearn.linear_model import LogisticRegression
            reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
            models.append(('logistic', reg))

            from sklearn.tree import DecisionTreeClassifier
            dtc = DecisionTreeClassifier()
            dtc.fit(X_train, y_train)
            models.append(('DecisionTreeClassifier', dtc))

            from sklearn.linear_model import SGDClassifier
            sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
            sgd_clf.fit(X_train, y_train)
            models.append(('SGDClassifier', sgd_clf))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)

            # Save the model for future use
            with open(model_path, 'wb') as f:
                pickle.dump((cv, classifier), f)

        except FileNotFoundError:
            print("Error: 'newdataset.csv' not found. Please ensure the data file exists.")
        except:
            print("An unexpected error occurred during data loading or training.")

    return cv, classifier

def predict(model_path, text):
    """Predicts the sentiment of a new text using the trained model.

    Args:
        model_path (str): Path to the pickled model file.
        text (str): The text for which to predict sentiment.

    Returns:
        str: The predicted sentiment label (e.g., "positive", "negative").
    """

    # Load the model and vectorizer
    cv, classifier = load_or_train_model(model_path)

    # Preprocess the text
    cleaned_text = clean_text(text)

    # Transform the text into a feature vector
    text_vec = cv.transform([cleaned_text])

    # Predict the sentiment
    prediction = classifier.predict(text_vec)[0]

    # Map the predicted label back to the original sentiment class
    labels = {0: 'negative', 1: 'positive', 2: 'anger'} 
    predicted_sentiment = labels[prediction]

    return predicted_sentiment


model_path = 'C:/Users/Chandu/Desktop/Sentiment analysis/my_sentiment_model.pkl'
load_or_train_model(model_path)
text = input("Enter the tweet message:")
# example :- text = "This movie was fantastic! I highly recommend it."
predicted_sentiment = predict(model_path, text)
print("Predicted sentiment:", predicted_sentiment)
