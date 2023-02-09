import re
import nltk
import pandas as pd
from langdetect import detect
from nltk.corpus import stopwords
from spacy.lang.it import Italian
from spacy.tokenizer import Tokenizer

nlp_ = Italian()
nltk.download('stopwords')

with open('other_words.txt', 'r') as f:
    other_words = f.readlines()
    other_words = [line.strip('\n') for line in other_words]

tokenizer = Tokenizer(nlp_.vocab)


def detect_language(sentence):
    # keep middle text
    start = int(round(len(sentence.split(' ')) / 3))
    end = start + int(round(start / 3))

    if start != 0 and end - start <= 3:
        end = start * 2
    # print(start, end)
    try:
        first_words = sentence.split(' ')[start:end]
        first_words = ' '.join(first_words)
        return detect(first_words) == 'it'
    except:
        return False


def detect_blacklist(sentence):
    blacklist = ['cookie', 'cookies', 'grazie attenzione', 'grazie', 'registrare', 'accedi', "inviato una email",
                 "riceverai un avviso quando", "registrata", "diritti riservati", "termini e condizioni", "clicca qui",
                 'accedere', 'email', "linvio non è andato a buon fine", 'posta elettronica', 'indirizzo di posta',
                 'newsletter', "attenzione linvio", "attenzione con linvio", "linvio non è andato a buon fine",
                 "privacy policy", "hai un account ", 'registrati', 'già registrato', 'trattamento dati',
                 'fini promozionali', 'profilazione', 'click', 'link', 'notifica']
    sentences = sentence.split('. ')
    sentences_new = []
    for sent in sentences:
        for word in blacklist:
            if word not in sent:
                continue
            else:
                break
        else:
            sentences_new.append(sent)

    sentences_new = ' '.join(sentences_new)
    return sentences_new


def detect_blacklist_sentences(sentence):
    blacklist_sentences = ["Utilizza solo immagini e fotografie rese disponibili a fini promozionali"]
    for sent in blacklist_sentences:
        if sent in sentence:
            return True

    return False


def remove_stopwords(sentence, stop_words, extrawords):
    doc = list(tokenizer(sentence))
    list_ = [str(token) for token in doc if
             str(token) not in stop_words and str(token) not in extrawords and len(str(token)) > 1 and 'http' not in str(
                 token) and 'www' not in str(token)]
    return ' '.join(list_)


def pre_preprocess_df(path_df):
    # Read data into papers
    print('Read dataframe....')
    papers = pd.read_parquet(path_df)
    print(f'      Original number of texts: {len(papers)}')

    print('Remove empty texts....')
    papers = papers[papers['txt'] != ''].reset_index(drop=True)
    print(f'      Only non-void texts: {len(papers)}')
    # keep only italian texts
    print('Remove non-italian texts....')
    papers = papers[papers['txt'].map(detect_language)].reset_index(drop=True)
    print(f'      Only italian texts: {len(papers)}')

    return papers


def preprocess_df(df_pre_preprocessed):
    # Remove records with some blacklist sentences

    print('Remove useless texts....')
    papers = df_pre_preprocessed[df_pre_preprocessed['txt'].map(detect_blacklist_sentences) == False].reset_index(
        drop=True)

    # Convert texts to lowercase
    print('Convert to lowercase....')
    papers['paper_text_processed'] = papers['txt'].apply(lambda x: x.lower())

    # Remove punctuation, symbols
    print('Removed digits, symbols, punctuation....')
    papers['paper_text_processed'] = papers['paper_text_processed'].apply(lambda x: re.sub(r"[^\w\s.]", '', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].apply(lambda x: re.sub(r'\d', '', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].apply(lambda x: re.sub(r' +', ' ', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].apply(lambda x: re.sub(r'_', '', x))

    # Remove sentence with blacklist words
    print('Removed blacklist sentences...')
    papers['paper_text_processed'] = papers['paper_text_processed'].apply(lambda x: detect_blacklist(x))

    # Remove .
    print('Removed dot...')
    papers['paper_text_processed'] = papers['paper_text_processed'].apply(lambda x: re.sub(r'\.', '', x))

    # Remove stopwords AND adverbs&co and words with 1 char
    print('Remove stopwords etc....')

    stop_words = stopwords.words('italian')
    stop_words.extend(['stato', 'stata', 'state', 'essere', 'de', 'the', 'of'])
    papers['paper_text_processed_nostopwords'] = papers['paper_text_processed'].apply(
        lambda x: remove_stopwords(x, stop_words, other_words))

    print('Remove too short text...')

    # Remove records with too short final text
    papers_noshort = papers[papers['paper_text_processed_nostopwords'].map(len) >= 150].reset_index(drop=True)

    print(f'Check language again...')
    papers_noshort_it = papers_noshort[papers_noshort['txt'].map(detect_language)].reset_index(drop=True)
    print('End')

    return papers_noshort_it
