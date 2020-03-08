import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
import pickle


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath = Filepath of the database where the data is stored
    
    OUTPUT:
    X = Returns the messages
    y = Returns the different categories
    category_names = Returns a list of the category columns
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df =  pd.read_sql_table('DisasterMessages', engine)
    X = df.message.values
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    '''
    INPUT:
    text = The message text
    
    OUTPUT: Returns the text after it has been tokenized, lower cased, striped, and lemmatized
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    OUTPUT: Returns the results of pipeline after the Grid Seach as been completed
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
            'clf__estimator__criterion': ['entropy', 'gini'],
            'clf__estimator__min_samples_split': [2, 4]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    INPUT:
    model = The model being used
    X_test = Test input features
    y_test = Test label features
    category_names = names of the categories
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    INPUT:
     model = The model being used
     model_filepath = File path for the model being used
    
    OUTPUT: Saves the model as a pickle file
    '''
    pickle.dump(model, open(model_filepath, "wb"))

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
