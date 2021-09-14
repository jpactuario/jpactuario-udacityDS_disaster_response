import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Load data from the database.

    Args:
        database_filepath (string): path of the database

    Returns:
        X (pd.DataFrame): features dataframe
        Y (pd.DataFrame): target dataframe
        category_names (list of string): list of target labels
    '''

    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.select_dtypes('number').drop(columns = 'id')
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenization function to be used in training.

    Args:
        text (str): string before tokenization

    Returns:
        tokens (list of str): tokenized and lemmatized list of words
    '''

    # Remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize into words
    words = word_tokenize(text)

    # Lemmatize words and also remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words
              if word not in stopwords.words("english")]

    return tokens

def build_model():
    '''
    Initializing both the classifier model (based on Random Forest using TF-IDF
    as the features) and the grid search of the `n_estimators` parameter in
    the Random Forest.

    Args:
        None

    Returns:
        cv (GridSearchCV class): initialized GridSearchCV
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    default_n_estimators = pipeline.get_params()['clf__estimator__n_estimators']
    lower_n_estimators = int(default_n_estimators * 0.9)
    higher_n_estimators = int(default_n_estimators * 1.1)
    parameters = {
        'clf__estimator__n_estimators': [
            lower_n_estimators,
            default_n_estimators,
            higher_n_estimators
        ]
    }
    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the accuracy / precision / recall / F-1 score of the models
    by means of Classification Report

    Args:
        model: model to be evaluated
        X_test (pd.DataFrame): testing set of the features
        Y_test (pd.DataFrame): ground truth of the testing set
        category_names (list of str): list of category names
    '''
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    print(classification_report(Y_test, Y_pred))

def save_model(model, model_filepath):
    '''
    Save the model into a pickle file.

    Args:
        model: model to be saved
        model_filepath (str): path and filename of the model pickle
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        # ADDED THIS LINE
        best_model = model.best_estimator_

        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
