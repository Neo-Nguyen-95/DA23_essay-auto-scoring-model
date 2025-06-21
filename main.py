#%% LIB
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

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

df = pd.read_csv('data/train.csv')

#%%skip_if True EDA 1

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
    
    row['text_length'] = count_text
    row['stopword_percent'] = percent_stopwords_text
    row['misspelled_percent'] = percent_misspelled
    
    return row

df = df.apply(describe_text, axis=1)




### TF-IDF (Term Frequency–Inverse Document Frequency)


### N-gram analysis




### Stop-word removal are not applied in this case-study



