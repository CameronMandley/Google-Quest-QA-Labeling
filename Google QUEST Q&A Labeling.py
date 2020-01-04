
# coding: utf-8

# In[1]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train = pd.read_csv('train.csv').dropna()
df_test = pd.read_csv('test.csv').dropna()
df_train.head()


# In[3]:


"""
Cleaning up the words
"""
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


# In[5]:


df_train['question_title'] = df_train['question_title'].apply(clean_text)
df_train['question_body'] = df_train['question_body'].apply(clean_text)
df_train['answer'] = df_train['answer'].apply(clean_text)


# In[6]:


target_labels = pd.read_csv('sample_submission.csv').columns
target_labels


# In[7]:


X = df_train['question_body']
y = df_train[target_labels]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state = 42)


# In[8]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
X_train = vectorizer.fit_transform(X_train).toarray()


# In[9]:


input_size = len(X_train)
output_size = 30


# In[38]:


from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
classifiers = np.array([])

for label in y_train.columns:
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(16), random_state=1, max_iter=2000)
    classifiers = np.append(classifiers, clf.fit(X_train,y_train[label]))


# In[53]:


i = 0

for label in y_train.columns:
    print(label, y_pred_trains[i])
    print('MSE %s' % sklearn.metrics.mean_squared_error(y_train[label], classifiers[i].predict(X_train), sample_weight=None, multioutput='uniform_average'))
    i = i+1


# In[66]:


y_train.columns


# In[57]:


df_test['question_title'] = df_test['question_title'].apply(clean_text)
df_test['question_body'] = df_test['question_body'].apply(clean_text)
df_test['answer'] = df_test['answer'].apply(clean_text)


# In[60]:


X_test = df_test['question_body']


# In[62]:


vectorizer = TfidfVectorizer (max_features=500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
X_test = vectorizer.fit_transform(X_test).toarray()


# In[55]:


predictions = pd.DataFrame()


# In[71]:


predictions['qa_id'] = df_test['qa_id']
predictions['question_asker_intent_understanding'] = classifiers[1].predict(X_test)
predictions['question_body_critical'] = classifiers[2].predict(X_test)
predictions['question_conversational'] = classifiers[3].predict(X_test)
predictions['question_expect_short_answer'] = classifiers[4].predict(X_test)
predictions['question_fact_seeking'] = classifiers[5].predict(X_test)
predictions['question_has_commonly_accepted_answer'] = classifiers[6].predict(X_test)
predictions['question_interestingness_others'] = classifiers[7].predict(X_test)
predictions['question_interestingness_self'] = classifiers[8].predict(X_test)
predictions['question_multi_intent'] = classifiers[9].predict(X_test)
predictions['question_not_really_a_question'] = classifiers[10].predict(X_test)
predictions['question_opinion_seeking'] = classifiers[11].predict(X_test)
predictions['question_type_choice'] = classifiers[12].predict(X_test)
predictions['question_type_compare'] = classifiers[13].predict(X_test)
predictions['question_type_consequence'] = classifiers[14].predict(X_test)
predictions['question_type_definition'] = classifiers[15].predict(X_test)
predictions['question_type_entity'] = classifiers[16].predict(X_test)
predictions['question_type_instructions'] = classifiers[17].predict(X_test)
predictions['question_type_procedure'] = classifiers[18].predict(X_test)
predictions['question_type_reason_explanation'] = classifiers[19].predict(X_test)
predictions['question_type_spelling'] = classifiers[20].predict(X_test)
predictions['question_well_written'] = classifiers[21].predict(X_test)
predictions['answer_helpful'] = classifiers[22].predict(X_test)
predictions['answer_level_of_information'] = classifiers[23].predict(X_test)
predictions['answer_plausible'] = classifiers[24].predict(X_test)
predictions['answer_relevance'] = classifiers[25].predict(X_test)
predictions['answer_satisfaction'] = classifiers[26].predict(X_test)
predictions['answer_type_instructions'] = classifiers[27].predict(X_test)
predictions['answer_type_procedure'] = classifiers[28].predict(X_test)
predictions['answer_type_reason_explanation'] = classifiers[29].predict(X_test)
predictions['answer_well_written'] = classifiers[30].predict(X_test)


# In[72]:


predictions


# In[73]:


predictions.to_csv(r'qa_results.csv', index=False)

