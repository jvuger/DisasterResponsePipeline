import sys

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    '''Loads data from provided database filepath'''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messag_categ', engine)
    X = df.message
    Y = df[df.columns[4:]]
    Y = Y.replace(to_replace=2, value=0)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''Tokenize, lemmatize, normalize case, remove leading/trailing white space'''
    
    #tokenize text
    tokens = text.split()
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    # iterate through each token
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", lemmatizer.lemmatize(tok).lower())
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Builds ML pipeline'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__min_samples_split': [2]
    }
    
    model = GridSearchCV(pipeline, param_grid = parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates the model by providing accuracy, precision and recall for each of the categories'''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, model_filepath):
    '''Exports the model to a .pkl file'''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''Loads the data. Builds, trains, evaluates and saves the model.'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()