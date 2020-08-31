import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    """
    Load Data
    
    Parameters:
    ----------------------------------------------------------------------------------------------
        database_filepath - path to SQLite db
        
    Results:
    ----------------------------------------------------------------------------------------------
        X - Features
        Y - Labels
        category_names - Categories of Disaster
    """
    engine= create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("df", engine)
    X=df['message']
    Y=df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text
    
    Parameters:
    ------------------------------------------------------------------------------------------
        text - sentence or message(english)
    Results:
    ------------------------------------------------------------------------------------------
        clean_tokens - tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        cleaned_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(cleaned_token)
        
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        """
        This function extracts the starting verb in the sentence which is now used as another feature on our table 
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
        This function builds the model 
        """
  pipeline = Pipeline([
         ('features', FeatureUnion([

             ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
             ])),

             ('starting_verb', StartingVerbExtractor()),
         ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
   
  parameters = {'features__text_pipeline__tfidf__use_idf': (True, False),
                  'features__text_pipeline__vect__max_df': (0.75, 1.0)
               }
         
  model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, verbose=2, cv=3)
  return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model
    """
    Y_pred = model.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    

    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
    
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
       print('Model Performance with Category: {}'.format(column))
       print(classification_report(Y_test[column],Y_pred[column]))


def save_model(model, model_filepath):
    """
    Save Model as Pickle file
    
    Parameters:
    -----------------------------------------------------------------
        model - GridSearchCV or Pipeline object
        model_filepath - destination path to save .pkl file
    
    """
    model_filepath = model_filepath
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