#%% LIB
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
nltk.download('words')
english_vocab = set(words.words())

from spellchecker import SpellChecker
from collections import Counter

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC

from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    )
from datasets import Dataset

import torch

df = pd.read_csv('data/train.csv')

#%% EDA 1

# df.head()

# df.info()

# fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))

# axes[0].bar(
#     df['score'].value_counts().sort_index().index,
#     df['score'].value_counts().sort_index()
#     )
# axes[0].set_title('Distribution of scores')
# axes[0].set_xlabel('Score')
# axes[0].set_ylabel('Count')

# axes[1].pie(
#     df['score'].value_counts(normalize=True).sort_index(),
#     labels=df['score'].value_counts(normalize=True).sort_index().index,
#     autopct='%1.1f%%'
#     )
# axes[1].set_title('Percentage of scores')

# plt.show()



#%% PREPROCESSING 1

# text = df.at[0, 'full_text']

# ### Normalization
# # Converting text to a standard form, e.g. lowercasing, removing punctuation, expanding contractions
# text = re.sub(r'[\.\?\!\,\:\;\"\'\xa0\$\d+]', ' ', text)
# text = text.lower()

# ### Tokenization -> nltk lib
# # splitting text into words or subwords
# tokenized_text = word_tokenize(text)
# count_text = len(tokenized_text)

# ### Stopword count
# stop_words = set(stopwords.words('english'))
# stopwords_text = [word for word in tokenized_text if word in stop_words]
# count_stopwords_text = len(stopwords_text)


# ### Stemming/lemmatization
# lemmatizer = WordNetLemmatizer()
# lemmatized_text = [lemmatizer.lemmatize(token) for token in tokenized_text]

# ### Count words
# Counter(lemmatized_text).most_common(10)

# ### Typo count
# spell = SpellChecker()
# misspelled = spell.unknown(lemmatized_text)
# corrected_spell = [spell.correction(word) for word in misspelled]  # Undetected word might be object/location name, I can filter it if SpellChecker cannot detect it.
# count_misspelled = len([word for word in corrected_spell if word])

### Stop-word removal are not applied in this case-study

#%% EDA 2

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def describe_text(row):
    text = row['full_text']
    
    # Normalization
    text = re.sub(r'[\.\?\!\,\:\;\"\'\xa0\$\d+]', ' ', text)
    text = text.lower()
    
    # Tokenization
    tokenized_text = word_tokenize(text)
    count_text = len(tokenized_text)
    
    # Stopword count/percentage
    stopwords_text = [word for word in tokenized_text if word in stop_words]
    count_stopwords_text = len(stopwords_text)
    percent_stopwords_text = (
        count_stopwords_text / count_text if count_text > 0 else 0
        )
    
    # Lemmatization
    
    lemmatized_text = [lemmatizer.lemmatize(token) for token in tokenized_text]
    
    # Typo count/percentage
    
    misspelled = spell.unknown(lemmatized_text)
    count_misspelled = len(misspelled)
    percent_misspelled = (
        count_misspelled / count_text if count_text > 0 else 0
        )
    
    # Collection of words (remove stop words)
    words = [word for word in lemmatized_text if word not in stop_words and word not in misspelled]
    
    # Result
    row['text_length'] = count_text
    row['stopword_percent'] = percent_stopwords_text
    row['misspelled_percent'] = percent_misspelled
    row['word_collection'] = words
    
    return row

df = df.apply(describe_text, axis=1)

# # Average table
# df.groupby('score')[['text_length', 'stopword_percent', 'misspelled_percent']].mean()

# # Text length plot
# sns.violinplot(
#     data=df,
#     x='text_length',
#     y='score',
#     orient='h',
#     split=True
#     )
# plt.title('Text\'s length distribution corresponding to score')
# plt.xlim(0, 1400)
# plt.show()

# # Stopword percentage plot
# sns.violinplot(
#     data=df,
#     x='stopword_percent',
#     y='score',
#     orient='h',
#     split=True
#     )
# plt.title('Distribution of percentage of stop words corresponding to score')
# plt.show()

# # Misspelled percentage plot
# sns.violinplot(
#     data=df,
#     x='misspelled_percent',
#     y='score',
#     orient='h',
#     split=True
#     )
# plt.title('Distribution of percentage of misspelled words  corresponding to score')
# plt.xlim(0, 0.1)
# plt.show()

#%% SCORING BY TEXT CLASSIFICATION
# # Classification with Naive Bayes, Logistic Regression, SVM

# ### Multinomial NB
# texts = [' '.join(words) for words in df['word_collection'].to_list()]
# labels = df['score'].to_list()

# X_train, X_val, y_train, y_val = train_test_split(
#     texts, labels, test_size=0.2, random_state=42, stratify=labels
#     )

# model_NB = make_pipeline(TfidfVectorizer(), MultinomialNB())
# model_NB.fit(X_train, y_train)
# prediction = model_NB.predict(X_val)
# print(f'Accuracy of Naive Bayes Classifier: {accuracy_score(y_val, prediction):.2f}')


# model_LR = make_pipeline(
#     TfidfVectorizer(), LogisticRegression(max_iter=500)
#     )
# model_LR.fit(X_train, y_train)
# prediction = model_LR.predict(X_val)
# print(f'Accuracy of Logistic Regression: {accuracy_score(y_val, prediction):.2f}')

# model_SVM = make_pipeline(TfidfVectorizer(), LinearSVC())
# model_SVM.fit(X_train, y_train)
# prediction = model_SVM.predict(X_val)
# print(f'Accuracy of SVM: {accuracy_score(y_val, prediction):.2f}')


# X_train, X_val, y_train, y_val = train_test_split(
#     df[['text_length', 'misspelled_percent']], df['score'], test_size=0.2, random_state=42, stratify=labels
#     )

# model_ridge = Ridge(alpha=1)
# model_ridge.fit(X_train, y_train)
# y_pred = model_ridge.predict(X_val)
# y_pred_rounded = [int(round(y, 0)) for y in y_pred]
# print(f'Accuracy of Ridge: {accuracy_score(y_val, y_pred_rounded):.2f}')

#%% SCORING BY FINE-TUNING BERT

# 1. Load tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

# 2. Convert the dataset
dataset = Dataset.from_pandas(df[['full_text', 'score']])
dataset = dataset.rename_column('score', 'label')

# 3. Tokenizzation function
def tokenizer_func(df):
    return tokenizer_bert(
        df['full_text'],  # Tokenize each essay
        truncation=True,  
        max_length=512,  # Truncate it to a maximum length of 512 tokens
        padding='max_length'  # all samples are the same length
        )

tokenized_dataset = dataset.map(tokenizer_func, batch_size=True)

# 4. Train - test split
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

# 5. Load BERT with a classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=6
    )  # scores 1–6


# 6. Training setup
training_args = TrainingArguments(
    output_dir="./bert-essay",  # save checkpoint
    evaluation_strategy="epoch",  # evaluate dataset after each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,  # regularize model
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # split your data for real use
)

# 7.Train
trainer.train()

# 8. Predict
preds = trainer.predict(eval_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)
true_labels = eval_dataset['label']

print(f"Accuracy of BERT: {accuracy_score(true_labels, pred_labels):.2f}")




