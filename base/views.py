from django.shortcuts import render, redirect
import numpy as np

# Create your views here.

import pickle

# Load the CountVectorizer (cv.pkl)
with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

# Load the RandomForest model (model.pkl)
with open('model.pkl', 'rb') as f:
    rf = pickle.load(f)


# decontractions of words
# these datas found on: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/19794953#19794953
contractions = {
  "ain't": "am not / are not / is not / has not / have not",
  "aren't": "are not / am not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he had / he would",
  "he'd've": "he would have",
  "he'll": "he shall / he will",
  "he'll've": "he shall have / he will have",
  "he's": "he has / he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how has / how is / how does",
  "I'd": "I had / I would",
  "I'd've": "I would have",
  "I'll": "I shall / I will",
  "I'll've": "I shall have / I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had / it would",
  "it'd've": "it would have",
  "it'll": "it shall / it will",
  "it'll've": "it shall have / it will have",
  "it's": "it has / it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she had / she would",
  "she'd've": "she would have",
  "she'll": "she shall / she will",
  "she'll've": "she shall have / she will have",
  "she's": "she has / she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so as / so is",
  "that'd": "that would / that had",
  "that'd've": "that would have",
  "that's": "that has / that is",
  "there'd": "there had / there would",
  "there'd've": "there would have",
  "there's": "there has / there is",
  "they'd": "they had / they would",
  "they'd've": "they would have",
  "they'll": "they shall / they will",
  "they'll've": "they shall have / they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had / we would",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what shall / what will",
  "what'll've": "what shall have / what will have",
  "what're": "what are",
  "what's": "what has / what is",
  "what've": "what have",
  "when's": "when has / when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where has / where is",
  "where've": "where have",
  "who'll": "who shall / who will",
  "who'll've": "who shall have / who will have",
  "who's": "who has / who is",
  "who've": "who have",
  "why's": "why has / why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had / you would",
  "you'd've": "you would have",
  "you'll": "you shall / you will",
  "you'll've": "you shall have / you will have",
  "you're": "you are",
  "you've": "you have"
}

# Basic text preprocessing

from bs4 import BeautifulSoup
import re
from nltk.stem import SnowballStemmer

def test_basic_preprocessing(question):

  # lower casing
  q = str(question).lower().strip()

  # html tags
  q = BeautifulSoup(q, features="html.parser").get_text()

  # replacing the certain characters with it's corresponding strings
  q = q.replace("%", "percent ")
  q = q.replace("$", "dollar ")
  q = q.replace("₹", "rupee ")
  q = q.replace("€", "euro ")
  q = q.replace("@", "at ")

  # The pattern `[math]` appearns in the datasets, so replacing to blank as: ""
  q = q.replace("[math]", "")

  # replacing numbers with it's corresponding string values
  q = q.replace(",000,000,000", "b ")
  q = q.replace(",000,000", "m ")
  q = q.replace(",000", "k ")
  q = re.sub(r"([0-9]+)000000000", r"\1b", q)
  q = re.sub(r"([0-9]+)000000", r"\1m", q)
  q = re.sub(r"([0-9]+)000", r"\1k", q)

  # decontraction of words
  q_decontracted = []
  for word in q.split():
    if word in contractions:
      word = contractions[word]

    q_decontracted.append(word)
  q = " ".join(q_decontracted)

  # replacing some decontracted words
  q = q.replace("'ve", "have")
  q = q.replace("n't", "not")
  q = q.replace("'re", "are")
  q = q.replace("'ll'", "will")

  # removing punctations
  pattern = re.compile("\W")
  q = re.sub(pattern, " ", q).strip()

  # stemming or lemmatizer
  snowball = SnowballStemmer("english")
  words = q.split()
  stemmed_word_list = [snowball.stem(word) for word in words]
  q = " ".join(stemmed_word_list)

  return q


def test_extra_preprocessing(q1, q2):

  extra_features = []

  # length of question1 & question2
  extra_features.append(len(str(q1)))
  extra_features.append(len(str(q2)))

  # words count of question1 & question2
  q1_word_count = len(q1.split(" "))
  q2_word_count = len(q2.split(" "))
  extra_features.append(q1_word_count)
  extra_features.append(q2_word_count)

  # total words
  total_words_count = q1_word_count + q2_word_count
  extra_features.append(total_words_count)

  # common words
  q1_words = set(map(lambda x: x.strip(), q1.split()))
  q2_words = set(map(lambda x: x.strip(), q2.split()))
  common_words_count = len(q1_words & q2_words)
  extra_features.append(common_words_count)

  # shared words
  shared_words = np.round((common_words_count/total_words_count), 2)
  extra_features.append(shared_words)

  return extra_features


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

def test_apply_token_features(q1, q2):
    
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8
    
    # split strings into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    
    # remove stopwords from each questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    # getting stop words from each questions
    q1_stop_words = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stop_words = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # get common non-stopwords from both questions
    common_words_count = len(q1_words.intersection(q2_words))
    
    # get common stop words count
    common_stopwords_count = len(q1_stop_words.intersection(q2_stop_words))
    
    # get commom token
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    # token features
    # common words count minimum
    token_features[0] = common_words_count/(min(len(q1_words), len(q2_words)) + SAFE_DIV)
    
    # common words count maximum
    token_features[1] = common_words_count/(max(len(q1_words), len(q2_words)) + SAFE_DIV)
    
    # common stop words count minimum
    token_features[2] = common_stopwords_count/(min(len(q1_stop_words), len(q2_stop_words)) + SAFE_DIV)
    
    # common stop words count minimum
    token_features[3] = common_stopwords_count/(max(len(q1_stop_words), len(q2_stop_words)) + SAFE_DIV)
    
    # common token count minimum
    token_features[4] = common_token_count/(min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # common token count minimum
    token_features[5] = common_token_count/(max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # last words of both questions, if same then 1 else 0
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # first words of both questions, if same then 1 else 0
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features


# length based features

import distance

def test_apply_length_feature(q1, q2):

  length_features = [0.0]*3

  # converting strings/questions into tokens
  q1_tokens = q1.lower().split()
  q2_tokens = q2.lower().split()

  if len(q1_tokens) == 0 or len(q2_tokens) == 0:
    return length_features

  # mean length
  length_features[0] = (len(q1_tokens) + len(q2_tokens))/2

  # absolute length features
  length_features[1] = abs(len(q1_tokens) - len(q2_tokens))

  # longest substring ratio
  strings = list(distance.lcsubstrings(q1, q2))
  if strings:
      longest_substring_length = len(strings[0])
  else:
      longest_substring_length = 0
  length_features[2] = longest_substring_length / (min(len(q1), len(q2)) + 1)

  return length_features


# fuzzy based features

from fuzzywuzzy import fuzz
def test_apply_fuzzy_features(q1, q2):

  fuzzy_features = [0.0]*4

  # fuzzy ratio
  fuzzy_features[0] = fuzz.QRatio(q1, q2)

  # partial ratio
  fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

  # token sort ratio
  fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

  # token set ratio
  fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

  return fuzzy_features


def index(request):

    def query_point_creator(q1, q2):

        input_query = []

        # basic preprocessing
        q1 = test_basic_preprocessing(q1)
        q2 = test_basic_preprocessing(q2)

        # extra 7 more features
        extra_features = test_extra_preprocessing(q1, q2)
        input_query.extend(extra_features)

        # token based features
        token_features = test_apply_token_features(q1, q2)
        input_query.extend(token_features)

        # length based features
        length_features = test_apply_length_feature(q1, q2)
        input_query.extend(length_features)

        # fuzzy features
        fuzzy_features = test_apply_fuzzy_features(q1, q2)
        input_query.extend(fuzzy_features)

        # bow features
        q1_bow = cv.transform([q1]).toarray()
        q2_bow = cv.transform([q2]).toarray()

        return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))
    
    result = None

    if request.method == 'POST':
        # Get values from the form fields
        q1 = request.POST.get('question1')
        q2 = request.POST.get('question2')

        # Process the form data (predict duplicate or not)
        if rf.predict(query_point_creator(q1, q2))[0] == 1:
            result = "Duplicate"
        else:
            result = "Not Duplicate"

        # Store the result in the session for this request only
        request.session['result'] = result

        # Redirect to the same page after processing the form to avoid resubmission
        return redirect('index')  # 'index' is the name of your view URL pattern

    # For GET requests or after POST redirection, fetch the result from the session and clear it
    result = request.session.get('result', None)

    # Optionally, clear the session result after it has been displayed once
    if result is not None:
        del request.session['result']  # Clear result from session to avoid stale data on refresh

    context = {
        'result': result
    }

    return render(request, "index.html", context)