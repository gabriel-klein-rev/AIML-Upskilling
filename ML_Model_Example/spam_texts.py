import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reading the data
df = pd.read_csv("./content/spam.csv",encoding='latin-1')
#print(df.head())

# Cleaning data
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df = df.rename(columns={'v1':'label','v2':'Text'})
df['label_enc'] = df['label'].map({'ham':0,'spam':1})
print(df.head())

sns.countplot(x=df['label'])
plt.show()


# Find average number of tokens in all sentences
avg_words_len=round(sum([len(i.split()) for i in df['Text']])/len(df['Text']))
print("Average word length:", avg_words_len)

# Finding Total no of unique words in corpus
s = set()
for sent in df['Text']:
  for word in sent.split():
    s.add(word)
total_words_length=len(s)
print("Total words:", total_words_length)

# Splitting data for Training and testing
from sklearn.model_selection import train_test_split

X, y = np.asanyarray(df['Text']), np.asanyarray(df['label_enc'])
new_df = pd.DataFrame({'Text': X, 'label': y})
X_train, X_test, y_train, y_test = train_test_split(
    new_df['Text'], new_df['label'], test_size=0.2, random_state=42)
print("Training data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# Building Baseline Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score

tfidf_vec = TfidfVectorizer().fit(X_train)
X_train_vec,X_test_vec = tfidf_vec.transform(X_train),tfidf_vec.transform(X_test)

baseline_model = MultinomialNB()
baseline_model.fit(X_train_vec,y_train)


# For compiling, fitting, and evaluating model
from sklearn.metrics import precision_score, recall_score, f1_score 

def compile_model(model): 
    ''' 
    simply compile the model with adam optimzer 
    '''
    model.compile(optimizer=keras.optimizers.Adam(), 
                loss=keras.losses.BinaryCrossentropy(), 
                metrics=['accuracy'])

def evaluate_model(model, X, y): 
    ''' 
    evaluate the model and returns accuracy, 
    precision, recall and f1-score 
    '''
    y_preds = np.round(model.predict(X)) 
    accuracy = accuracy_score(y, y_preds) 
    precision = precision_score(y, y_preds) 
    recall = recall_score(y, y_preds) 
    f1 = f1_score(y, y_preds) 

    model_results_dict = {'accuracy': accuracy, 
                        'precision': precision, 
                        'recall': recall, 
                        'f1-score': f1} 

    return model_results_dict 

def fit_model(model, epochs, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test):
  '''
  fit the model with given epochs, train
  and test data
  '''
  # Check if validation data is provided
  if X_test is not None and y_test is not None:
      history = model.fit(X_train,
                      y_train,
                      epochs=epochs,
                      validation_data=(X_test, y_test)) #Removed validation steps argument
  else:
      # Handle case where validation data is not provided
      history = model.fit(X_train, 
                      y_train,
                      epochs=epochs)
  return history


baseline_model_results = evaluate_model(baseline_model, X_test_vec, y_test)

total_results = pd.DataFrame({'MultinomialNB Model':baseline_model_results}).transpose()

print(total_results)
