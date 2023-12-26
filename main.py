import pandas as pd
import os
import nltk
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from pymystem3 import Mystem
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

mystem = Mystem()
stop_words = set(stopwords.words('russian'))
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()


def preprocess_text(text: str):
    """Function gets text and removes stopwords and patterns

    Args:
        text (str): text for preprocess 

    Returns:
        List[str]: preprocessed text
    """
    words = re.sub(patterns, ' ', text)
    tokens = []
    for token in words.split():
        if token and token not in stop_words:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    text = " ".join(tokens)
    tokens = word_tokenize(text)
    return tokens

def preprocess_text_only_A(text: str) :
    """Function gets text, lemmatize them and removes all word without adjective and adverb

    Args:
        text (str): text for preprocess

    Returns:
        List[str]: preprocessed text
    """
    tokens = mystem.lemmatize(str(text))
    tokens = [token for token in tokens if token not in stop_words]
    text = " ".join(tokens)
    words = nltk.word_tokenize(text)
    functors_pos = {'A=m', 'ADV'}
    res = [word for word, pos in nltk.pos_tag(words, lang='rus')
           if pos not in functors_pos]
    return res

def count_words(text : str):
    """Function counts information about the number of words in a cell
    
    Args:
        text (str) : 
    Return:
        len (List[str]): Ammount words in list with words from text
    """
    words = nltk.word_tokenize(text)
    words_only = [word for word in words if re.match(r"^\w+$", word)]
    return len(words_only)

def read_review(file_path : str):
    """Function returns the text of reviews without spaces
    
    Args:
        file_path (str) : path to the file to be read 
    Return:
        review_text(str) : ready text of reviews without spaces
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if len(lines) >= 4:
            review_text = ''.join(line.rstrip() for line in lines[3:])
            return review_text.strip()
        else:
            return None
        

def filter_by_word_count(dataframe : pd.DataFrame, word_count : int):
    """Function that selects strings for which the value in the column 
    with the number of words is less than or equal to the specified value.

    Args:
        dataframe (pd.DataFrame): DataFrame with text information
        word_count (int): number of words
    Return:
        pd.DataFrame: DataFrame with a word count exceeding max count
    """
    return dataframe[dataframe['Количество слов'] <= word_count]


def filter_by_stars(dataframe : pd.DataFrame, label : str):
    """Function that sorts the dataset by the given DataFrame label.
    
    Args:
        dataframe (pd.DataFrame): DataFrame with text information
        label (str): DataFrame class label
    Return:
        pd.DataFrame: DataFrame with a number of stars
    """
    dataframe['Количество звёзд'] = dataframe['Количество звёзд'].astype(str)
    if label in ['1', '2', '3', '4', '5']:
        dataframe[dataframe['Количество звёзд'] == label]
    elif label == "other":
        dataframe[~dataframe['Количество звёзд'].isin(['1', '2', '3', '4', '5'])]
    else:
        raise ValueError("Неверное значение для метки класса. Допустимые значения: от 1 до 5 и 'other'")
    return dataframe


def plot_word_histogram(dataframe : pd.DataFrame, label : str):
    """ Function make plot with matplotlib x-axe is a frequency of occurrence, y-axe is a word

    Args: 
        dataframe (pd.DataFrame): DataFrame with text information
        label (str): DataFrame class label
    """
    dataframe['Количество звёзд'] = dataframe['Количество звёзд'].astype(str)
    if label == "other":
        text_block = " ".join(dataframe[~dataframe['Количество звёзд'].isin(['1', '2', '3', '4', '5'])]['Текст рецензии'])
    else:
        text_block = " ".join(dataframe[dataframe['Количество звёзд'] == label]['Текст рецензии'])

    
    tokens = preprocess_text_only_A(preprocess_text(text_block))
    

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

    word_freq = Counter(lemmatized_tokens)

    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:20]) 
    
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_word_freq.keys(), sorted_word_freq.values())
    plt.xlabel('Слова')
    plt.ylabel('Частота встречаемости')
    plt.title('Гистограмма слов для метки класса ' + str(label))
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    for i, (word, freq) in enumerate(sorted_word_freq.items()):
        plt.text(i, freq + 0.5, str(freq), ha='center', va='bottom', rotation=90, fontsize=8)

    plt.show()


    
root_folder = 'data'

data = []

for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                review_text = read_review(file_path)
                data.append({'Количество звёзд': folder, 'Текст рецензии': review_text})

df = pd.DataFrame(data)

df.fillna('ПУСТОЙ ОТЗЫВ', inplace=True)

df['Количество слов'] = df['Текст рецензии'].apply(count_words)

print(df.head())

numeric_info1 = df[['Количество звёзд']].describe()
print(numeric_info1)

numeric_info2 = df[['Количество слов']].describe()
print(numeric_info2)

df.to_csv('data.csv', index=False)

filtered_df = filter_by_word_count(df,20)
print(filtered_df)
filtered_df.to_csv('data_of_filtered_df_by_words.csv', index=False)

filterd_df_by_stars = filter_by_stars(df, "1")
print(filterd_df_by_stars)
filterd_df_by_stars.to_csv('data_filterd_df_by_stars.csv', index=False)

grouped = df.groupby('Количество звёзд')['Количество слов'].agg(['max', 'min', 'mean'])
print(grouped)

plot_word_histogram(df, '2')