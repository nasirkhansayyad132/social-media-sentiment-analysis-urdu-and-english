
from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.http import HttpResponse
# from django.contrib.auth.models import User

from django.contrib.auth.models import User, UserManager


from .fbpagescripper import fetch_page_posts

import tensorflow as tf
import numpy as np
import re
from tweepy import *
import sys
from django.contrib.auth.models import User
import re
from django.db import IntegrityError
import pandas as pd
import time
from django.contrib import messages
# from facebook_page_scraper import Facebook_scraper
import json
import nltk
from urduhack import normalize
from urduhack.normalization import normalize_characters
from django.shortcuts import render, redirect

from django.shortcuts import render
import tensorflow as tf
import numpy as np
import re
from tweepy import *
import sys
import tweepy
import pandas as pd
import time
from django.contrib import messages
from facebook_page_scraper import Facebook_scraper
import json
import nltk
from urduhack import normalize
from urduhack.normalization import normalize_characters
# from .fbpagescripper import fetch_page_posts
import itertools
import json
import random

from neo4j import GraphDatabase

from nltk.tokenize import sent_tokenize

from urduhack.tokenization import sentence_tokenizer


import tensorflow as tf
import re
import string
import numpy as np
from tensorflow.keras.models import load_model
from transformers import TFBertModel
from transformers import BertTokenizer
from tensorflow.keras.optimizers import Adam

cons_key = ' 0JlLfk2DhC4U5obUNIAc4q0a9'
cons_secret = 'apcrcAJB5Iyhdu6inDSnp5SiQTRBkrK96gde3axcTUBtuhiX2h'
acc_token = '876524568176885761-UurIHeMqrwo1Db3OcsvEiqi5rqBPJSO'
acc_secret = 'xUEGzDIpc5EtdHhTu7wkMkuH0QrLJPJjMY1j91e1CsBZJ'

auth = tweepy.OAuth1UserHandler(
    cons_key, cons_secret, acc_token, acc_secret
)
api = tweepy.API(auth)


# (1) Authentication Function
def get_twitter_auth():
    try:
        consumer_key = cons_key
        consumer_secret = cons_secret
        access_token = acc_token
        access_secret = acc_secret
    except KeyError:
        sys.stderr.write("Twitter Environment Variable not Set\n")
        sys.exit(1)

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth

# # (2) Function to get the Twitter client


def get_twitter_client():
    auth = get_twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client

# (3) Function to create the final dataframe


def get_tweets_from_user(twitter_user_name, page_limit=1, count_tweet=10):
    client = get_twitter_client()
    all_tweets = []
    api = tweepy.API(auth)
    for page in Cursor(client.user_timeline, screen_name=twitter_user_name, count=count_tweet, include_rts=False, tweet_mode='extended').pages(page_limit):
        for tweet in page:
            parsed_tweet = {}
            if tweet.lang == "en":
                filteredText = tweet.full_text.split("https")[0]
                parsed_tweet['text'] = filteredText
                all_tweets.append(parsed_tweet)

    df = pd.DataFrame(all_tweets)
    df = df.drop_duplicates("text", keep='first')
    return df


def base(request):

    return render(request, 'base.html')


def validate_username(username):
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]+$"
    return re.match(pattern, username)


def validate_email(email):
    pattern = r"^\w+[\w.-]*@\w+[\w.-]+\.\w+$"
    return re.match(pattern, email)


def validate_password(password):
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$"
    return re.match(pattern, password)

from django.contrib.auth.models import User
def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        # Username validation
        if not any(char.islower() for char in uname):
            error_msg_username = "Username should contain at least one lowercase letter."
            return render(request, 'signup.html', {'error_msg_username': error_msg_username})

        if not any(char.isupper() for char in uname):
            error_msg_username = "Username should contain at least one uppercase letter."
            return render(request, 'signup.html', {'error_msg_username': error_msg_username})

        if not any(char.isdigit() for char in uname):
            error_msg_username = "Username should contain at least one digit."
            return render(request, 'signup.html', {'error_msg_username': error_msg_username})

        # Email validation
        if len(email) < 8 or '@' not in email:
            error_msg_email = "Invalid email address."
            return render(request, 'signup.html', {'error_msg_email': error_msg_email})

        # Password validation
        if len(pass1) < 8 or not any(char.islower() for char in pass1) or not any(char.isupper() for char in pass1) or not any(char.isdigit() for char in pass1):
            error_msg_password = "Password should be at least 8 characters long and contain at least one lowercase letter, one uppercase letter, and one digit."
            return render(request, 'signup.html', {'error_msg_password': error_msg_password})

        if pass1 != pass2:
            error_msg_confirm_password = "Your password and confirm password are not the same."
            return render(request, 'signup.html', {'error_msg_confirm_password': error_msg_confirm_password})

        try:
            my_user = User.objects.create_user(
                username=uname, email=email, password=pass1)
            my_user.save()
            return redirect('login')
        except IntegrityError:
            error_msg_username = "Username already exists. Please choose a different username."
            return render(request, 'signup.html', {'error_msg_username': error_msg_username})

    return render(request, 'signup.html')


def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('pass')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('base')
        else:
            messages.error(request, 'Username or password is incorrect.')
    return render(request, 'login.html')


def LogoutPage(request):
    logout(request)
    return redirect('login')


def blog(request):

    return render(request, 'blog.html')


def facebook(request):
    pos = 0
    neg = 0
    neu = 0
    tweets_array = []
    tweets_result_array = []
    context = {}

    if request.method == "POST":
        if request.POST.get('pageid'):
            acountname = request.POST.get('pageid')
            print("page",acountname)
            myTweets = fetch_page_posts(acountname)
            print("my tweets:",myTweets)
            
            model_path = "/home/nasir/FYP/After30%/CNNwith4/cnnenglishmodel.tflite"
            context = process_single_user_fb(myTweets,acountname, model_path)
            print("context n:", context)
    return render(request, 'facebook.html', context)

def bert_analysis_fb(tweets):
    # Load the BERT model and tokenizer from the transformers library
    vocab = np.load(
        '/home/nasir/FYP/After30%/CNNwith4/bertvocab.npy', allow_pickle=True).item()
    custom_objects = {'TFBertModel': TFBertModel}
    bert_model_path = '/home/nasir/FYP/After30%/CNNwith4/bertcls3.h5'
    model = load_model(bert_model_path, custom_objects=custom_objects)

    # Compile the model with the same optimizer and loss function used during training
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab=vocab)

    # Test tweets
    # tweets = [
    #     ["Pakistan Air Force (PAF) FT-7PG trainer aircraft, while recovering from a routine operational training mission, crashed during landing at #Peshawar Air Base. Rescue operation is in progress."],
    #     ["You are not in this world to live up to other people's expectations, nor should you feel the world must live up to yours   F Perl."],
    #     ["Congrats to Mian sb, Maryam bibi &amp; Captain sb. NAB failed again. Intent of political victimisation proved again. NAB/Niazi unholy alliance is getting exposed by the day. Whole process was meant to push PTI forward, but unfortunately it pushed our country backward. God bless Pak!"],
    #     ["Meeting with Chairman Imran khan on current political scenario & upcoming jalsa's of PTI in Sindh."],
    #     ["what the government is doing its not good for economy"],
    #     ["government is successful to provide help for people"],
    #     ["May be this will be your last birthday  prime minister nawaz shareef"]
    # ]

    labels = ['negative', 'neutral', 'positive']

    # Import the necessary NLTK module

    tweets_dict = {}

    # Set the maximum sentence length
    max_length = 152

    # Iterate over the list of lists of tweets
    i=0
    # for tweet_list in tweets:
    # Initialize the tweet dictionary
    tweet_dict = {}

    # Initialize dictionaries to store label counts and scores
    label_counts = {}
    label_scores = {}

    # Iterate over the tweets in the current list
    for tweet in tweets:
        # Tokenize the tweet into sentences
        sentences = sent_tokenize(tweet)

        # Iterate over the sentences
        for k, sentence in enumerate(sentences):
            # Preprocess the sentence
            cleaned_sentence = clean_str_bert(sentence)

            # Tokenize and pad the sentence
            tokens = tokenizer(cleaned_sentence, truncation=True,
                                padding="max_length", max_length=max_length, return_tensors="tf")
            sentence_input_ids = tokens["input_ids"]
            sentence_attention_masks = tokens["attention_mask"]

            # Classify the sentence
            sentence_predictions = model.predict(
                [sentence_input_ids, sentence_attention_masks])
            sentence_max_score = np.max(sentence_predictions)
            sentence_predicted_index = np.argmax(sentence_predictions)
            sentence_predicted_label = labels[sentence_predicted_index]

            # Update label counts and scores
            if sentence_predicted_label in label_counts:
                label_counts[sentence_predicted_label] += 1
                if sentence_predicted_label in label_scores and sentence_predicted_label == tweet_dict.get(f"sentence{k+1}", [None, 0, None])[2]:
                    label_scores[sentence_predicted_label] = max(
                        label_scores[sentence_predicted_label], sentence_max_score)
                else:
                    label_scores[sentence_predicted_label] = sentence_max_score
            else:
                label_counts[sentence_predicted_label] = 1
                label_scores[sentence_predicted_label] = sentence_max_score

            # Add the sentence information to the tweet dictionary
            tweet_dict[f"sentence{k+1}"] = [cleaned_sentence,
                                            sentence_max_score, sentence_predicted_label]

    # Determine the predicted label for the tweet
    repeated_labels = [label for label,
                        count in label_counts.items() if count > 1]
    if repeated_labels:
        max_repeated_label = max(repeated_labels, key=lambda label: (
            label_counts[label], label_scores[label]))
        tweet_predicted_label = max_repeated_label
    else:
        tweet_predicted_label = max(label_counts, key=label_counts.get)

    # Add the tweet dictionary and predicted label to the tweets dictionary
    tweets_dict[f"Post {i+1}"] = tweet_dict
    tweets_dict[f"Post {i+1}"]["predicted_label"] = tweet_predicted_label

    # Print the resulting tweets dictionary
    print(tweets_dict)

    return tweets_dict


def process_single_user_fb(fb,twitter_user_name, model_path):
    pos = 0
    neg = 0
    neu = 0
    suggestions = 0
    # products_list=[]
    # # Get tweets from the user
    # # myTweets = get_tweets_from_user(twitter_user_name)
    # # products_list = myTweets.values.tolist()
    # if fb:
    #     products_list = fb
    # else:
    #     products_list = [
    #         ["The government's efforts in improving healthcare facilities are commendable. They have taken significant steps to ensure accessible healthcare for all citizens."],
    #         ["Corruption is deeply rooted in the government system and needs to be eradicated. It hampers progress and undermines public trust."],
    #         ["The education sector needs significant reforms to ensure quality education for all. Investing in education is crucial for the country's future."],
    #         ["The government's economic policies have stabilized the country's economy and attracted foreign investments. This has resulted in job creation and improved living standards."],
    #         ["The law and order situation has improved under the current government, making the streets safer. Efforts to curb crime and enhance security have yielded positive results."],
    #         ["There is a lack of transparency in government contracts, which raises concerns about accountability. Steps should be taken to promote transparency and prevent corruption."],
    #         ["The government should prioritize environmental conservation by implementing stricter regulations. Protecting the environment is crucial for sustainable development."],
    #         ["The government's efforts in improving healthcare facilities are commendable. They have taken significant steps to ensure accessible healthcare for all citizens."],
    #         ["Corruption is deeply rooted in the government system and needs to be eradicated. It hampers progress and undermines public trust."],
    #         ["The education sector needs significant reforms to ensure quality education for all. Investing in education is crucial for the country's future."]

    #     ]

    tweet_dict = {}
    tweet_index =0
    for tweet in fb:
        sentences = sent_tokenize(tweet)
        tweet_results = {}

        for sentence_index, sentence in enumerate(sentences):
            cleaned_sentence = clean_str(sentence)
            input_data = prepare_input(cleaned_sentence)
            results = predict_sentence(model_path, input_data)
            max_result = max(results, key=lambda x: x['score'])
            class_label = max_result['class_label']
            score = max_result['score']
            tweet_results[f"sentence{sentence_index+1}"] = [
                cleaned_sentence, score, class_label]

        tweet_dict[f"Post {tweet_index+1}"] = tweet_results

        if class_label == 'Positive':
            pos += 1
        elif class_label == 'Negative':
            neg += 1
        else:
            neu += 1

        total_tweets = pos + neg + neu
        avg = ""
        if pos >= neg and pos >= neu:
            avg = "Positive"
        elif neg >= pos and neg >= neu:
            avg = "Negative"
        else:
            avg = "Neutral"

        pos_avg = round((pos / total_tweets) * 100, 2)
        neg_avg = round((neg / total_tweets) * 100, 2)
        neu_avg = round((neu / total_tweets) * 100, 2)

    for i in range(1, 101):
        tweet_key = f"Post {i}"
        if tweet_key in tweet_dict:
            tweet = tweet_dict[tweet_key]
            max_score = float('-inf')  # Initialize with a very low score
            predicted_label = None

            for j in range(1, len(tweet) + 1):
                sentence_key = f"sentence{j}"
                if sentence_key in tweet:
                    sentence_value = tweet[sentence_key]
                    sentence_text = sentence_value[0]
                    score = sentence_value[1]
                    class_label = sentence_value[2]

                    if score > max_score:
                        max_score = score
                        predicted_label = class_label
            tweet["predicted_label"] = predicted_label

    bert_result = bert_analysis_fb(fb)
    pos_tweets = 0
    neg_tweets = 0
    neu_tweets = 0

    # Iterate through the dictionary
    for tweet in bert_result.values():
        predicted_label = tweet['predicted_label']
        if predicted_label == 'positive':
            pos_tweets += 1
        elif predicted_label == 'negative':
            neg_tweets += 1
        elif predicted_label == 'neutral':
            neu_tweets += 1

    avgb = ""
    if pos_tweets >= neg_tweets and pos_tweets >= neu_tweets:
        avgb = "Positive"
    elif neg_tweets >= pos_tweets and neg_tweets >= neu_tweets:
        avgb = "Negative"
    else:
        avgb = "Neutral"

    total = pos_tweets + neg_tweets + neu_tweets

    pos_avgb = round((pos_tweets / total) * 100, 2)
    neg_avgb = round((neg_tweets / total) * 100, 2)
    neu_avgb = round((neu_tweets / total) * 100, 2)

    context = {
        'tweets_dict': tweet_dict,
        'bert_result': bert_result,

        'total_tweets': len(tweet_dict),
        'avg': avg,
        'pos_avg': pos_avg,
        'neg_avg': neg_avg,
        'neu_avg': neu_avg,
        'pos_tweets': pos,
        'neg_tweets': neg,
        'neu_tweets': neu,

        'avgb': avgb,
        'pos_avgb': pos_avgb,
        'neg_avgb': neg_avgb,
        'neu_avgb': neu_avgb,
        'pos_tweetsb': pos_tweets,
        'neg_tweetsb': neg_tweets,
        'neu_tweetsb': neu_tweets,
        'acountname': "user.name",
        'username': "twitter_user_name"
    }

    print("context", context)

    return context




def clean_str_bert(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace('\r', '').replace('\n', ' ').replace(
        '\n', ' ').lower()  # remove \n and \r and lowercase
    # remove links and mentions
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_list = string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text.strip().lower()



def classifyTweet(request):
    max_length = 152
    labels = ['negative', 'neutral', 'positive']
    tweet_dict = {}

        # Initialize dictionaries to store label counts and scores
    label_counts = {}
    label_scores = {}
    context = {}
    if request.method == "POST":
        inputtweet = request.POST.get('tweet')
        language = request.POST.get('language')

        print("Selected Language:", language)  # Add this line to check the selected language

        if language == "English":
            print(inputtweet)
            sentences = sent_tokenize(inputtweet)
            
            vocab = np.load(
            '/home/nasir/FYP/After30%/CNNwith4/bertvocab.npy', allow_pickle=True).item()
            custom_objects = {'TFBertModel': TFBertModel}
            bert_model_path = '/home/nasir/FYP/After30%/CNNwith4/bertcls3.h5'
            model = load_model(bert_model_path, custom_objects=custom_objects)

            # Compile the model with the same optimizer and loss function used during training
            # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab=vocab)

            # Iterate over the sentences
            for k, sentence in enumerate(sentences):
                # Preprocess the sentence
                cleaned_sentence = clean_str_bert(sentence)

                # Tokenize and pad the sentence
                tokens = tokenizer(cleaned_sentence, truncation=True,
                                   padding="max_length", max_length=max_length, return_tensors="tf")
                sentence_input_ids = tokens["input_ids"]
                sentence_attention_masks = tokens["attention_mask"]

                # Classify the sentence
                sentence_predictions = model.predict(
                    [sentence_input_ids, sentence_attention_masks])
                sentence_max_score = np.max(sentence_predictions)
                sentence_predicted_index = np.argmax(sentence_predictions)
                sentence_predicted_label = labels[sentence_predicted_index]

                # Update label counts and scores
                if sentence_predicted_label in label_counts:
                    label_counts[sentence_predicted_label] += 1
                    if sentence_predicted_label in label_scores and sentence_predicted_label == tweet_dict.get(f"sentence{k+1}", [None, 0, None])[2]:
                        label_scores[sentence_predicted_label] = max(
                            label_scores[sentence_predicted_label], sentence_max_score)
                    else:
                        label_scores[sentence_predicted_label] = sentence_max_score
                else:
                    label_counts[sentence_predicted_label] = 1
                    label_scores[sentence_predicted_label] = sentence_max_score

                # Add the sentence information to the tweet dictionary
                tweet_dict[f"sentence{k+1}"] = [cleaned_sentence,
                                                sentence_max_score, sentence_predicted_label]
            print(tweet_dict)

        # elif language == "Urdu":
        #     vocab = np.load(
        # '/home/nasir/FYP/After30%/CNNwith4/bertvocaburdu.npy', allow_pickle=True).item()
        #     custom_objects = {'TFBertModel': TFBertModel}
        #     bert_model_path = '/home/nasir/FYP/After30%/CNNwith4/bert_urdu_cls3.h5'
        #     model = load_model(bert_model_path, custom_objects=custom_objects)

        #     # Compile the model with the same optimizer and loss function used during training
        #     # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

        #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab=vocab)

        #     # Test tweets
        #     # tweets = [
        # #         ["حکومت کا کارروائیوں پر میں بہت خوش ہوں۔. حکومت نے معیشت کو بہتر بنانے کیلئے اقدامات اٹھائے ہیں۔" ],
        # #         ["حکومت کے فیصلوں سے میں بہت ناراض ہوں-۔ حکومت کا معیشتی حالات پر کوئی نظرانداز نہیں کیا جا سکتا۔"],
        # #         ["پاکستان کی حکومت نے بجٹ کی تشکیل کے حوالے سے بہت اچھی کارروائی کی ہے۔- حکومت کی برقراری میں کچھ توجہ کی ضرورت ہے۔"],
        #     # #    ]

        #     labels = ['negative', 'neutral', 'positive']

        #     tweets_dict = {}

        #     # Set the maximum sentence length
        #     max_length = 128
        #     label_counts = {}
        #     label_scores = {}

        #     sentences = sentence_tokenizer(inputtweet)

        #     # Iterate over the sentences
        #     for k, sentence in enumerate(sentences):
        #         # Preprocess the sentence
        #         cleaned_sentence = clean_str_urdu(sentence)

        #         # Tokenize and pad the sentence
        #         tokens = tokenizer(cleaned_sentence, truncation=True, padding="max_length", max_length=max_length, return_tensors="tf")
        #         sentence_input_ids = tokens["input_ids"]
        #         sentence_attention_masks = tokens["attention_mask"]

        #         # Classify the sentence
        #         sentence_predictions = model.predict([sentence_input_ids, sentence_attention_masks])
        #         sentence_max_score = np.max(sentence_predictions)
        #         sentence_predicted_index = np.argmax(sentence_predictions)
        #         sentence_predicted_label = labels[sentence_predicted_index]

        #         # Update label counts and scores
        #         if sentence_predicted_label in label_counts:
        #             label_counts[sentence_predicted_label] += 1
        #             if sentence_predicted_label in label_scores and sentence_predicted_label == tweet_dict.get(f"sentence{k+1}", [None, 0, None])[2]:
        #                 label_scores[sentence_predicted_label] = max(label_scores[sentence_predicted_label], sentence_max_score)
        #             else:
        #                 label_scores[sentence_predicted_label] = sentence_max_score
        #         else:
        #             label_counts[sentence_predicted_label] = 1
        #             label_scores[sentence_predicted_label] = sentence_max_score

        #         # Add the sentence information to the tweet dictionary
        #         tweet_dict[f"sentence{k+1}"] = [cleaned_sentence, sentence_max_score, sentence_predicted_label]
        #     print("urdu:",tweet_dict)

        elif language == "Urdu":
            model_path = "/home/nasir/FYP/After30%/CNNwith4/converted_model_urdu3.tflite"
            # tweet_text = tweet
            # print("tweet text:", tweet_text)
            sentences = sentence_tokenizer(inputtweet)
            tweet_results = {}
            print("sentences:", sentences)
            for sentence_index, sentence in enumerate(sentences):
                print("sentence:", sentence)
                cleaned_sentence = clean_str_urdu(sentence)
                print("cleaned sentence: ", cleaned_sentence)
                input_data = prepare_input_urdu(cleaned_sentence)
                print("input data: ", input_data)
                results = predict_sentence_urdu(model_path, input_data)
                max_result = max(results, key=lambda x: x['score'])
                class_label = max_result['class_label']
                score = max_result['score']
                tweet_results[f"sentence{sentence_index+1}"] = [
                    cleaned_sentence, score, class_label]
            tweet_dict = tweet_results
        else:
            tweet_dict = {"Language not supported"}

        context = {
            'result': tweet_dict
        }
        return render(request, 'base.html', context)

    return render(request, 'base.html')

   

def facebookurdu(request):

    return render(request, 'facebookurdu.html')


def followers(request):
    print("\n\nin followers\n\n")

    pos = 0
    neg = 0
    neu = 0
    avg_array = []
    following_names_array = []
    pos_array = []
    neu_array = []
    neg_array = []
    pol_array = []
    foll_screen_names = []
    if request.method == "POST":
        if request.POST.get('usernametwitter'):
            acountname = request.POST.get('usernametwitter')

            products_list = getFollowings(acountname)
            avg = []
            print("my all tweets", products_list)
            print("\n")

            for j in range(0, len(products_list)):

                for i in range(0, len(products_list[j])):
                    if i == 0:
                        user = api.get_user(screen_name=products_list[j][i])

                        following_names_array.append(user.name)
                        foll_screen_names.append(products_list[j][i])
                        print("product i of j 0 names", products_list[j][i])
                        print("\n")
                        continue
                    print("product i of j", products_list[j][i])
                    print("\n")
                    # result = predict_tweet(products_list[j][i])
                    # print("result:",result)
                    try:
                        print("in before try: ")
                        print("\n")
                        result = predict_tweet(products_list[j][i])
                        print("in try result: ", result)
                        print("\n")
                        if result == 0:
                            pos = pos + 1
                        elif result == 1:
                            neg = neg + 1
                        else:
                            neu = neu + 1
                            time.sleep(10)
                    except:
                        print("in pass result: ", pos, neg, neu)
                        print("\n")
                        pass
                if pos >= neg and pos >= neu:
                    avg.append("Positive")
                    pol_array.append(0)
                elif neg >= pos and neg >= neu:
                    avg.append("Negative")
                    pol_array.append(1)
                else:
                    avg.append("Neutral")
                    pol_array.append(2)

                total_tweets = pos + neg + neu
                # if(total_tweets > 0):
                pos_array.append(round((pos / total_tweets)*100, 2))
                neg_array.append(round((neg / total_tweets) * 100, 2))
                neu_array.append(round((neu / total_tweets)*100, 2))
                # else:
                #     pos_array.append(0)
                #     neg_array.append(0)
                #     neu_array.append(0)

                if pos_array[j] >= neg_array[j] and pos_array[j] >= neu_array[j]:
                    avg_array.append(pos_array[j])
                elif neg_array[j] >= pos_array[j] and neg_array[j] >= neu_array[j]:
                    avg_array.append(neg_array[j])
                else:
                    avg_array.append(neu_array[j])
            following_names_array
            pos_user = 0
            neu_user = 0
            neg_user = 0

            for k in range(0, len(avg)):
                if avg[k] == 'Positive':
                    pos_user = pos_user + 1
                elif avg[k] == 'Negative':
                    neg_user = neg_user + 1
                else:
                    neu_user = neu_user + 1
            total_followings = pos_user + neg_user + neu_user
            avg_total = ""
            if pos_user >= neg_user and pos_user >= neu_user:
                avg_total = "Positive"
            elif neg_user >= pos_user and neg_user >= neu_user:
                avg_total = "Negative"
            else:
                avg_total = "Neutral"
            print(neu_array)
            mydata = createJson(foll_screen_names,
                                following_names_array, avg_array, pol_array)
            jsonData = json.dumps(mydata)
            print(mydata)
            if (mydata):
                data_vailable = True
            # print(avg)
            user = api.get_user(screen_name=acountname)

            context2 = {
                'following_result': zip(following_names_array, pos_array, neg_array, neu_array, avg),
                'data_vailable': data_vailable,
                'positive_user': round((pos_user / total_followings)*100, 2),
                'negative_user': round((neg_user / total_followings)*100, 2),
                'neutral_user': round((neu_user / total_followings)*100, 2),
                'total_users': total_followings,
                'no_pos_user': pos_user,
                'no_neg_user': neg_user,
                'no_neu_user': neu_user,
                'acountname': user.name,
                'avg_total': avg_total,
                'graph': jsonData,
            }

            return render(request, 'following.html', context2)
    return render(request, 'following.html')


def getFollowings(screen_name, followings=5, cnt=3):
    print("\n\nin getfollowings\n\n")

    client = get_twitter_client()
    api = tweepy.API(auth)
    AllTweets = []
    for friend in tweepy.Cursor(client.get_friends, screen_name=screen_name).items(followings):
        tweets = client.user_timeline(screen_name=friend.screen_name,
                                      # 200 is the maximum allowed count
                                      count=cnt,
                                      include_rts=False,
                                      # Necessary to keep full_text
                                      # otherwise only the first 140 words are extracted
                                      tweet_mode='extended'
                                      )
        # print(friend.screen_name)
        parsed_tweets = []
        parsed_tweets.append(f"{friend.screen_name}")
        for tweet in tweets:
            if tweet.lang == "en":
                filteredText = tweet.full_text.split("https")[0]

                parsed_tweets.append(filteredText)
                # print(parsed_tweets)
        AllTweets.append(parsed_tweets)

    return AllTweets
# =======================================================================


def createJson(foll_screen_names, f_names, avg, pol):
    print("\n\nin json\n\n")

    client = get_twitter_client()
    data = {}
    at_list = []
    for i in range(0, len(f_names)):
        at_list.append({"username": foll_screen_names[i], "character": f_names[i],
                       "id": i, "influence": avg[i], "zone": random.randint(0, 6), "pol": pol[i]})

    data["nodes"] = at_list
    print("Data user name", data["nodes"])
    edge_list = []
    for s, t in itertools.combinations(foll_screen_names, 2):
        friendship = client.get_friendship(source_screen_name=s,
                                           target_screen_name=t)
        target_name = s
        # Find the object with the specified id
        target_obj = next(
            filter(lambda obj: obj["username"] == target_name, data["nodes"]), None)
        print("Target Object", target_obj)
        # Access the value of the names attribute
        if target_obj:
            ids = target_obj["id"]
            print("Target ids", ids)

        target_name = t
        # Find the object with the specified id
        target_obj = next(
            filter(lambda obj: obj["username"] == target_name, data['nodes']), None)

        # Access the value of the names attribute
        if target_obj:
            idt = target_obj["id"]

        if friendship[0].following:
            edge_list.append({
                "source": ids,
                "target": idt,
                "weight": 6
            })
        if friendship[1].following:
            edge_list.append({
                "source": idt,
                "target": ids,
                "weight": 6
            })
    data["links"] = edge_list

    return data


def singleuser(request):
    pos = 0
    neg = 0
    neu = 0
    suggestions = 0
    tweets_array = []
    tweets_result_array = []
    context = {}
    model_path = "/home/nasir/FYP/After30%/CNNwith4/cnnenglishmodel.tflite"

    if request.method == "POST":

        print("my value", request.POST.get('dropdown_value'))

        if request.POST.get('usernametwitter'):
            if request.POST.get('dropdown_value') == 'twitter':
                acountname = request.POST.get('usernametwitter')
                context = process_single_user(acountname, model_path)
                save_to_db(context)
                print("context n:", context)
    return render(request, 'singleuser.html', context)


# Clean the string
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# # Pad sentences


def pad_sentences_new(sentences, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = 152 - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

# Prepare input for prediction


def prepare_input(text):
    x_text = [clean_str(text)]
    x_text = [s.split(" ") for s in x_text]
    sentences_padded = pad_sentences_new(x_text)
    vocabulary = np.load(
        r'/home/nasir/FYP/After30%/CNNwith4//data1-vocab.npy', allow_pickle=True).item()
    cleaned = []
    for word in sentences_padded:
        for word2 in word:
            if word2 in vocabulary:
                cleaned.append(word2)
            else:
                sentences_padded[0].remove(word2)
                sentences_padded[0].append("<PAD/>")
                cleaned.append("<PAD/>")
    x2 = np.array([[vocabulary.get(word2, 0) for word2 in cleaned]
                  for cleaned in sentences_padded])
    return x2.astype(np.float32)

# Perform the prediction


def predict_sentence(model_path, input_data):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(
        input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['Positive', 'Negative', 'Neutral']
    results = []

    for i in range(len(classes)):
        result = {
            'class_label': classes[i],
            'score': output_data[0][i]
        }
        results.append(result)

    return results

# Function to process a single user


def process_single_user(twitter_user_name, model_path):
    pos = 0
    neg = 0
    neu = 0
    suggestions = 0

    # Get tweets from the user
    # myTweets = get_tweets_from_user(twitter_user_name)
    # products_list = myTweets.values.tolist()
    products_list = [
        ['Today all govt. servants were relaxed. There was no pressure on them. According to PmlN establishment was busy in london'],
        ['Taking into account the history of the partys u-turns, PTI will back off from its todays announcement of dissolving assemblies within a week. '],
        ['PM Imran going to Saudia and China to beg billions of dollars before the visit of IMF delegation to get a better deal. But it would be adding up 30 billion dollars in debts. Great achievement in breaking the begging bowl?'],
        ['IK Cabinet looks like a recipe to accommodate partners rather to put right man at the right job.'],
        ["The government's efforts in improving healthcare facilities are commendable. They have taken significant steps to ensure accessible healthcare for all citizens."],
        ["Corruption is deeply rooted in the government system and needs to be eradicated. It hampers progress and undermines public trust."],
        ["The education sector needs significant reforms to ensure quality education for all. Investing in education is crucial for the country's future."],
        ["The government's economic policies have stabilized the country's economy and attracted foreign investments. This has resulted in job creation and improved living standards."],
        ["The law and order situation has improved under the current government, making the streets safer. Efforts to curb crime and enhance security have yielded positive results."],
        ["There is a lack of transparency in government contracts, which raises concerns about accountability. Steps should be taken to promote transparency and prevent corruption."],
        ["The government should prioritize environmental conservation by implementing stricter regulations. Protecting the environment is crucial for sustainable development."],
        ["The government's efforts in improving healthcare facilities are commendable. They have taken significant steps to ensure accessible healthcare for all citizens."],
        ["Corruption is deeply rooted in the government system and needs to be eradicated. It hampers progress and undermines public trust."],
        ["The education sector needs significant reforms to ensure quality education for all. Investing in education is crucial for the country's future."]

    ]

    print("len product", len(products_list))
    print(products_list[1][0])
    tweet_dict = {}

    for tweet_index, tweet in enumerate(products_list):
        tweet_text = tweet[0]
        sentences = sent_tokenize(tweet_text)
        tweet_results = {}

        for sentence_index, sentence in enumerate(sentences):
            cleaned_sentence = clean_str(sentence)
            input_data = prepare_input(cleaned_sentence)
            results = predict_sentence(model_path, input_data)
            max_result = max(results, key=lambda x: x['score'])
            class_label = max_result['class_label']
            score = max_result['score']
            tweet_results[f"sentence{sentence_index+1}"] = [
                cleaned_sentence, score, class_label]

        tweet_dict[f"tweet{tweet_index+1}"] = tweet_results

        if class_label == 'Positive':
            pos += 1
        elif class_label == 'Negative':
            neg += 1
        else:
            neu += 1

        total_tweets = pos + neg + neu + suggestions
        avg = ""
        if pos >= neg and pos >= neu:
            avg = "Positive"
        elif neg >= pos and neg >= neu:
            avg = "Negative"
        else:
            avg = "Neutral"

        pos_avg = round((pos / total_tweets) * 100, 2)
        neg_avg = round((neg / total_tweets) * 100, 2)
        neu_avg = round((neu / total_tweets) * 100, 2)

    for i in range(1, 101):
        tweet_key = f"tweet{i}"
        if tweet_key in tweet_dict:
            tweet = tweet_dict[tweet_key]
            max_score = float('-inf')  # Initialize with a very low score
            predicted_label = None

            for j in range(1, len(tweet) + 1):
                sentence_key = f"sentence{j}"
                if sentence_key in tweet:
                    sentence_value = tweet[sentence_key]
                    sentence_text = sentence_value[0]
                    score = sentence_value[1]
                    class_label = sentence_value[2]

                    if score > max_score:
                        max_score = score
                        predicted_label = class_label
            tweet["predicted_label"] = predicted_label

    bert_result = bert_analysis(products_list)
    pos_tweets = 0
    neg_tweets = 0
    neu_tweets = 0

    # Iterate through the dictionary
    for tweet in bert_result.values():
        predicted_label = tweet['predicted_label']
        if predicted_label == 'positive':
            pos_tweets += 1
        elif predicted_label == 'negative':
            neg_tweets += 1
        elif predicted_label == 'neutral':
            neu_tweets += 1

    avgb = ""
    if pos_tweets >= neg_tweets and pos_tweets >= neu_tweets:
        avgb = "Positive"
    elif neg_tweets >= pos_tweets and neg_tweets >= neu_tweets:
        avgb = "Negative"
    else:
        avgb = "Neutral"

    total = pos_tweets + neg_tweets + neu_tweets

    pos_avgb = round((pos_tweets / total) * 100, 2)
    neg_avgb = round((neg_tweets / total) * 100, 2)
    neu_avgb = round((neu_tweets / total) * 100, 2)

    context = {
        'tweets_dict': tweet_dict,
        'bert_result': bert_result,

        'total_tweets': len(tweet_dict),
        'avg': avg,
        'pos_avg': pos_avg,
        'neg_avg': neg_avg,
        'neu_avg': neu_avg,
        'pos_tweets': pos,
        'neg_tweets': neg,
        'neu_tweets': neu,

        'avgb': avgb,
        'pos_avgb': pos_avgb,
        'neg_avgb': neg_avgb,
        'neu_avgb': neu_avgb,
        'pos_tweetsb': pos_tweets,
        'neg_tweetsb': neg_tweets,
        'neu_tweetsb': neu_tweets,
        'acountname': "user.name",
        'username': "twitter_user_name"
    }

    print("context", context)

    return context


def bert_analysis(tweets):
    # Load the BERT model and tokenizer from the transformers library
    vocab = np.load(
        '/home/nasir/FYP/After30%/CNNwith4/bertvocab.npy', allow_pickle=True).item()
    custom_objects = {'TFBertModel': TFBertModel}
    bert_model_path = '/home/nasir/FYP/After30%/CNNwith4/bertcls3.h5'
    model = load_model(bert_model_path, custom_objects=custom_objects)

    # Compile the model with the same optimizer and loss function used during training
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab=vocab)

    # Test tweets
    # tweets = [
    #     ["Pakistan Air Force (PAF) FT-7PG trainer aircraft, while recovering from a routine operational training mission, crashed during landing at #Peshawar Air Base. Rescue operation is in progress."],
    #     ["You are not in this world to live up to other people's expectations, nor should you feel the world must live up to yours   F Perl."],
    #     ["Congrats to Mian sb, Maryam bibi &amp; Captain sb. NAB failed again. Intent of political victimisation proved again. NAB/Niazi unholy alliance is getting exposed by the day. Whole process was meant to push PTI forward, but unfortunately it pushed our country backward. God bless Pak!"],
    #     ["Meeting with Chairman Imran khan on current political scenario & upcoming jalsa's of PTI in Sindh."],
    #     ["what the government is doing its not good for economy"],
    #     ["government is successful to provide help for people"],
    #     ["May be this will be your last birthday  prime minister nawaz shareef"]
    # ]

    labels = ['negative', 'neutral', 'positive']

    # Import the necessary NLTK module

    tweets_dict = {}

    # Set the maximum sentence length
    max_length = 152

    # Iterate over the list of lists of tweets
    for i, tweet_list in enumerate(tweets):
        # Initialize the tweet dictionary
        tweet_dict = {}

        # Initialize dictionaries to store label counts and scores
        label_counts = {}
        label_scores = {}

        # Iterate over the tweets in the current list
        for j, tweet in enumerate(tweet_list):
            # Tokenize the tweet into sentences
            sentences = sent_tokenize(tweet)

            # Iterate over the sentences
            for k, sentence in enumerate(sentences):
                # Preprocess the sentence
                cleaned_sentence = clean_str_bert(sentence)

                # Tokenize and pad the sentence
                tokens = tokenizer(cleaned_sentence, truncation=True,
                                   padding="max_length", max_length=max_length, return_tensors="tf")
                sentence_input_ids = tokens["input_ids"]
                sentence_attention_masks = tokens["attention_mask"]

                # Classify the sentence
                sentence_predictions = model.predict(
                    [sentence_input_ids, sentence_attention_masks])
                sentence_max_score = np.max(sentence_predictions)
                sentence_predicted_index = np.argmax(sentence_predictions)
                sentence_predicted_label = labels[sentence_predicted_index]

                # Update label counts and scores
                if sentence_predicted_label in label_counts:
                    label_counts[sentence_predicted_label] += 1
                    if sentence_predicted_label in label_scores and sentence_predicted_label == tweet_dict.get(f"sentence{k+1}", [None, 0, None])[2]:
                        label_scores[sentence_predicted_label] = max(
                            label_scores[sentence_predicted_label], sentence_max_score)
                    else:
                        label_scores[sentence_predicted_label] = sentence_max_score
                else:
                    label_counts[sentence_predicted_label] = 1
                    label_scores[sentence_predicted_label] = sentence_max_score

                # Add the sentence information to the tweet dictionary
                tweet_dict[f"sentence{k+1}"] = [cleaned_sentence,
                                                sentence_max_score, sentence_predicted_label]

        # Determine the predicted label for the tweet
        repeated_labels = [label for label,
                           count in label_counts.items() if count > 1]
        if repeated_labels:
            max_repeated_label = max(repeated_labels, key=lambda label: (
                label_counts[label], label_scores[label]))
            tweet_predicted_label = max_repeated_label
        else:
            tweet_predicted_label = max(label_counts, key=label_counts.get)

        # Add the tweet dictionary and predicted label to the tweets dictionary
        tweets_dict[f"tweet{i+1}"] = tweet_dict
        tweets_dict[f"tweet{i+1}"]["predicted_label"] = tweet_predicted_label

    # Print the resulting tweets dictionary
    print(tweets_dict)

    return tweets_dict


def clean_str_bert(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace('\r', '').replace('\n', ' ').replace(
        '\n', ' ').lower()  # remove \n and \r and lowercase
    # remove links and mentions
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_list = string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text.strip().lower()


driver = GraphDatabase.driver("bolt://localhost:7689", auth=("nasir", "mydb.132"))


# Database work
# ==============================================================================

# Define a function to retrieve data from Neo4j database

driver = GraphDatabase.driver("bolt://localhost:7689", auth=("nasir", "mydb.132"))

from neo4j import GraphDatabase
from datetime import datetime, timedelta
# Neo4j connection details


def create_user( tweets, date_saved):
    
    tweets_dict = tweets["tweets_dict"]
    bert_result = tweets["bert_result"]
    
    date_savedc = date_saved
    # Extract data from the BERT node properties
    avgb = tweets["avgb"]
    pos_avgb = tweets["pos_avgb"]
    neg_avgb = tweets["neg_avgb"]
    neu_avgb = tweets["neu_avgb"]
    pos_tweetsb = tweets["pos_tweetsb"]
    neg_tweetsb = tweets["neg_tweetsb"]
    neu_tweetsb = tweets["neu_tweetsb"]

    # Extract data from the account
    total_tweets = tweets["total_tweets"]
    username= tweets["username"]
    pos_avg = tweets["pos_avg"]
    neg_avg = tweets["neg_avg"]
    neu_avg = tweets["neu_avg"]
    pos_tweets = tweets["pos_tweets"]
    neg_tweets = tweets["neg_tweets"]
    neu_tweets = tweets["neu_tweets"]
    avg = tweets["avg"]
    
    with driver.session(database="twittersentiment") as session:
        session.run("create (u:User {username: $username, date_saved: $date_saved})", username=username, date_saved=date_saved)

        cnn_node = session.run(
            """
            MATCH (u:User {username: $username, date_saved:$date_saved})
            create (u)-[:POSTED]->(c:CNN)
            SET c += {
                total_tweets: $total_tweets,
                avg: $avg,
                pos_avg: $pos_avg,
                neg_avg: $neg_avg,
                neu_avg: $neu_avg,
                pos_tweets: $pos_tweets,
                neg_tweets: $neg_tweets,
                neu_tweets: $neu_tweets,
                date_savedc: $date_savedc
            }
            RETURN c
            """,
            username=username, date_saved = date_saved,
            total_tweets=total_tweets,
            avg=avg,
            pos_avg=pos_avg,
            neg_avg=neg_avg,
            neu_avg=neu_avg,
            pos_tweets=pos_tweets,
            neg_tweets=neg_tweets,
            neu_tweets=neu_tweets,
            date_savedc=date_savedc
        ).single()[0]

        bert_node = session.run(
            """
            MATCH (u:User {username: $username,  date_saved:$date_saved})
            create (u)-[:POSTED]->(b:BERT)
            SET b += {
                avgb: $avgb,
                pos_avgb: $pos_avgb,
                neg_avgb: $neg_avgb,
                neu_avgb: $neu_avgb,
                pos_tweetsb: $pos_tweetsb,
                neg_tweetsb: $neg_tweetsb,
                neu_tweetsb: $neu_tweetsb,
                date_savedc: $date_savedc
            }
            RETURN b
            """,
            username=username, date_saved = date_saved,
            avgb=avgb,
            pos_avgb=pos_avgb,
            neg_avgb=neg_avgb,
            neu_avgb=neu_avgb,
            pos_tweetsb=pos_tweetsb,
            neg_tweetsb=neg_tweetsb,
            neu_tweetsb=neu_tweetsb,
            date_savedc = date_savedc
        ).single()[0]

        for tweet_id, tweet_data in tweets_dict.items():
            tweet_node = session.run(
                """
                MATCH (c:CNN)
                WHERE id(c) = $cnn_node_id
                create (c)-[:CONTAINS]->(t:Tweet {
                    tweet_id: $tweet_id,
                    predicted_label: $predicted_label,
                    date_saved:$date_saved
                })
                RETURN t
                """,
                cnn_node_id=cnn_node.id,
                tweet_id=tweet_id,
                predicted_label=tweet_data["predicted_label"],
                date_saved=date_saved
            ).single()[0]

            for sentence_id, sentence_data in tweet_data.items():
                if sentence_id != "predicted_label":
                    session.run(
                        """
                        MATCH (t:Tweet)
                        WHERE id(t) = $tweet_node_id
                        CREATE (t)-[:HAS_SENTENCE]->(:Sentence {
                            text: $text,
                            score: $score,
                            sentiment: $sentiment
                        })
                        """,
                        tweet_node_id=tweet_node.id,
                        text=sentence_data[0],
                        score=sentence_data[1],
                        sentiment=sentence_data[2],
                    )

        for tweet_id, tweet_data in bert_result.items():
            tweet_node = session.run(
                """
                MATCH (b:BERT)
                WHERE id(b) = $bert_node_id
                create (b)-[:CONTAINS]->(t:Tweet {
                    tweet_id: $tweet_id,
                    predicted_label: $predicted_label,
                    date_saved:$date_saved
            })
            RETURN t
            """,
            bert_node_id=bert_node.id,
            tweet_id=tweet_id,
            predicted_label=tweet_data["predicted_label"],
            date_saved=date_saved
            ).single()[0]

            for sentence_id, sentence_data in tweet_data.items():
                if sentence_id != "predicted_label":
                    session.run(
                        """
                        MATCH (t:Tweet)
                        WHERE id(t) = $tweet_node_id
                        CREATE (t)-[:HAS_SENTENCE]->(:Sentence {
                            text: $text,
                            score: $score,
                            sentiment: $sentiment
                        })
                        """,
                        tweet_node_id=tweet_node.id,
                        text=sentence_data[0],
                        score=sentence_data[1],
                        sentiment=sentence_data[2],
                    )

    # Close the Neo4j driver
    driver.close()



def save_to_db(tweets):
    date_saved = datetime.now().strftime("%Y-%m-%d")
    two_weeks_ago = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    
    # tweets = context_data
    username = tweets['username']
    with driver.session(database="twittersentiment") as session:
        result = session.run(
        """
        MATCH (u:User {username: $username})
        RETURN u.username as username, u.date_saved as date_saved, u.id as id ORDER BY u.date_saved DESC
        """,
        username=username ).single()

        if result:
                print("Node exists")
                print(result)
                existing_date_saved = result['date_saved']
                print("exting date",existing_date_saved)
                print(two_weeks_ago)
                if existing_date_saved >= two_weeks_ago:
                    print("Overwriting data")
                    existing_user = result['username']
                    existing_user_id = result['id']
                    session.run(
                        """
                    MATCH (u:User {username: $username})-[:POSTED]->(:CNN)-[:CONTAINS]->(t:Tweet)-[:HAS_SENTENCE]->(s:Sentence)
                        WHERE id(u) = $user_id and u.date_saved = $date_saved
                        DETACH DELETE u,t,s
                        """,username = username,
                        user_id=existing_user_id, date_saved = date_saved
                    )
                    return "Data overwritten"
                else:
                    print("Not overwriting data, new record saved")
                    
                    create_user(tweets,date_saved)
                    return "Not overwriting data, new record saved"
        else:
            print("new node with data created successfully")
            create_user(tweets,date_saved)
            return "new node with data created successfully"


def connect_to_db():
    driver = GraphDatabase.driver(
        "bolt://localhost:7689", auth=("nasir", "mydb.132"))
    return driver

# -------------------------------------------------------------------


def history(request):
    context = {}
    if request.method == 'POST':
        duration = request.POST.get('time_range')
        # Rest of your code
        acountname = request.POST.get('usernametwitter')
        # Call the retrieve_data function with the selected duration
        
        context = retrieve_history(acountname, duration)
        print("my context: ",context)
        # Pass the context to the template for rendering
    return render(request, 'history.html', context)

    
def retrieve_result(username,date_saved):
    
    context = {}
    with driver.session(database="twittersentiment") as session:
        cnn_data = session.run(
            """
            MATCH (u:User {username: $username,date_saved:$date_saved})-[:POSTED]->(c:CNN)
                    RETURN c
            """,
            username=username,date_saved = date_saved
        ).single()[0]
        # date = session.run(
        #     "match (u:User {username: $usrn"
        #     )
        # Store CNN node data in the context
        print("cnn data:",cnn_data)
        context= {
            "total_tweets": cnn_data["total_tweets"],
            "avg": cnn_data["avg"],
            "pos_avg": cnn_data["pos_avg"],
            "neg_avg": cnn_data["neg_avg"],
            "neu_avg": cnn_data["neu_avg"],
            "pos_tweets": cnn_data["pos_tweets"],
            "neg_tweets": cnn_data["neg_tweets"],
            "neu_tweets": cnn_data["neu_tweets"],
            "date_saved":cnn_data["date_savedc"]
        }
    
        # Retrieve tweet data from CNN
        cnn_tweet_data = session.run(
            """
            MATCH (c:CNN)-[:CONTAINS]->(t:Tweet )
            where datetime(c.date_savedc) = datetime($date_saved)
            RETURN t.tweet_id AS tweet_id, t.predicted_label AS predicted_label
            """,
            date_saved = date_saved
        ).data()
        i = 0
        context["CNN"] = context["CNN"] if "CNN" in context else {}
        for tweet in cnn_tweet_data:
            tweet_id = tweet["tweet_id"]
            predicted_label = tweet["predicted_label"]
            cnn_sentence_data = session.run(
               """
    MATCH (u:User {username: $username})-[:POSTED]->(c:CNN)-[:CONTAINS]->(t:Tweet {tweet_id: $tweet_id})-[:HAS_SENTENCE]->(s:Sentence)
    where datetime(u.date_saved) = datetime($date_saved)
    RETURN u.username AS username, s.text AS text, s.score AS score, s.sentiment AS sentiment
    """
    ,
                tweet_id=tweet_id,username=username,date_saved=date_saved
            ).data()
    
            tweet_data = {}
            k=0
            for sentence in cnn_sentence_data:
                cleaned_sentence = sentence["text"]
                sentence_max_score = sentence["score"]
                sentence_predicted_label = sentence["sentiment"]
                tweet_data[f"sentence{k+1}"] = [cleaned_sentence, sentence_max_score, sentence_predicted_label]
                k += 1
    
            tweet_data["predicted_label"] = predicted_label
            context["CNN"][f"tweet{i+1}"] = tweet_data
            i +=1
        print("\n\nCNN data",context,"\n\n")
        # Retrieve BERT node data
        bert_data = session.run(
            """
            MATCH (u:User {username: $username,date_saved:$date_saved})-[:POSTED]->(b:BERT)
            
            RETURN b
            """,
            username=username, date_saved=date_saved
        ).single()[0]
    
        # Store BERT node data in the context
        print("\n\nbert data:",bert_data['avgb'])
        context ["avgb"]= bert_data["avgb"]
        context["pos_avgb"]= bert_data["pos_avgb"]
        context["neg_avgb"]= bert_data["neg_avgb"]
        context["neu_avgb"]= bert_data["neu_avgb"]
        context["pos_tweetsb"]= bert_data["pos_tweetsb"]
        context["neg_tweetsb"]= bert_data["neg_tweetsb"]
        context["neu_tweetsb"]= bert_data["neu_tweetsb"]
        context["date_saved"]=cnn_data["date_savedc"]

        
        print("\n\nnew context ",context)
        # Retrieve tweet data from BERT
        bert_tweet_data = session.run(
            """
            MATCH (b:BERT)-[:CONTAINS]->(t:Tweet)
            where datetime(b.date_savedc) = datetime($date_saved)
            RETURN t.tweet_id AS tweet_id, t.predicted_label AS predicted_label
            """,date_saved= date_saved
        ).data()
        context['BERT'] = {}
        i = 0
        for tweet in bert_tweet_data:
            tweet_id = tweet["tweet_id"]
            predicted_label = tweet["predicted_label"]
            bert_sentence_data = session.run(
                """
     MATCH (u:User {username: $username})-[:POSTED]->(b:BERT)-[:CONTAINS]->(t:Tweet {tweet_id: $tweet_id})-[:HAS_SENTENCE]->(s:Sentence)
     where datetime(u.date_saved) = datetime($date_saved)
     RETURN u.username AS username, s.text AS text, s.score AS score, s.sentiment AS sentiment
     """
     ,
                 username=username, tweet_id=tweet_id,date_saved= date_saved
            ).data()
    
            tweet_data = {}
            k=0
            for sentence in bert_sentence_data:
                cleaned_sentence = sentence["text"]
                sentence_max_score = sentence["score"]
                sentence_predicted_label = sentence["sentiment"]
                tweet_data[f"sentence{k+1}"] = [cleaned_sentence, sentence_max_score, sentence_predicted_label]
                k += 1
    
            tweet_data["predicted_label"] = predicted_label
            context["BERT"][f"tweet{i+1}"] = tweet_data
            i +=1
    
    # Close the Neo4j driver
    driver.close()
    print("\n\nCNN data",context["CNN"],"\n\n")

    return context

    
def retrieve_history(username,date):
    # Connect to the database
    driver = GraphDatabase.driver("bolt://localhost:7689", auth=("nasir", "mydb.132"))


    with driver.session(database="twittersentiment") as session:
    # Retrieve CNN node data
    
        current_date = datetime.now()
        one_month_ago = current_date - timedelta(days=30)
        two_weeks_ago = current_date - timedelta(days=14)
        two_months_ago = current_date - timedelta(days=60)
        date_saved = ""
        context = None
        # if (date=="2weeks")
        # date = "one_month"
        try:
            if date == "2weeks":
                result = session.run("""
                    MATCH (u:User {username: $username})-[:POSTED]->(c:CNN)
                    WHERE datetime(u.date_saved) > datetime($two_weeks_ago)
                    RETURN c, u.date_saved
                """, username=username, two_weeks_ago=two_weeks_ago)

                for record in result:
                    c = record["c"]
                    date_saved = record["u.date_saved"]

                context = retrieve_result(username, date_saved)
                
            elif date == "one_month":
                result = session.run("""
                    MATCH (u:User {username: $username})-[:POSTED]->(c:CNN)
                    WHERE datetime(u.date_saved) < datetime($two_weeks_ago) and datetime(u.date_saved) >= datetime($one_month_ago)
                    RETURN c, u.date_saved
                """, username=username, one_month_ago=one_month_ago, two_weeks_ago=two_weeks_ago)

                for record in result:
                    c = record["c"]
                    date_saved = record["u.date_saved"]

                context = retrieve_result(username, date_saved)

            else:
                result = session.run("""
                    MATCH (u:User {username: $username})-[:POSTED]->(c:CNN)
                    WHERE datetime(u.date_saved) < datetime($two_months_ago)
                    RETURN c, u.date_saved
                """, username=username, two_months_ago=two_months_ago)

                for record in result:
                    c = record["c"]
                    date_saved = record["u.date_saved"]

                context = retrieve_result(username, date_saved)

            if context:
                return context
            else:
                context = {"err": "No Record found!"}
                return context

        except Exception as e:
            # Handle the exception here
            error_message = str(e)
            context = {"err": "No Record Found"}
            return context

# res = retrieve_data("one_month", "2weeks")
# print(res)
# ------------------------------------------------------------------------------

# (1). Athentication Function


def get_twitter_auth():
    """
    @return:
        - the authentification to Twitter
    """
    try:
        consumer_key = cons_key
        consumer_secret = cons_secret
        access_token = acc_token
        access_secret = acc_secret

    except KeyError:
        sys.stderr.write("Twitter Environment Variable not Set\n")
        sys.exit(1)

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    return auth


# (2). Client function to access the authentication API
def get_twitter_client():
    """
    @return:
        - the client to access the authentification API
    """
    auth = get_twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client


# (3). Function creating final dataframe
def get_tweets_from_user(twitter_user_name, page_limit=1, count_tweet=15):
    """
    @params:
        - twitter_user_name: the twitter username of a user (company, etc.)
        - page_limit: the total number of pages (max=16)
        - count_tweet: maximum number to be retrieved from a page

    @return
        - all the tweets from the user twitter_user_name
    """
    client = get_twitter_client()

    all_tweets = []
    api = tweepy.API(auth)
    for page in Cursor(client.user_timeline,
                       screen_name=twitter_user_name,
                       count=count_tweet, tweet_mode='extended').pages(page_limit):
        for tweet in page:
            parsed_tweet = {}
            # parsed_tweet['date'] = tweet.created_at
            # parsed_tweet['author'] = tweet.user.name
            # parsed_tweet['twitter_name'] = tweet.user.screen_name

            # the screen name of the user
            # screen_name = "geeksforgeeks"
            # fetching the user
            # fetching the ID
            if tweet.lang == "en":
                filteredText = tweet.full_text.split("https")[0]
                parsed_tweet['text'] = filteredText
                # parsed_tweet['number_of_likes'] = tweet.favorite_count
                # parsed_tweet['number_of_retweets'] = tweet.retweet_count

                all_tweets.append(parsed_tweet)

    # Create dataframe
    df = pd.DataFrame(all_tweets)

    # Revome duplicates if there are any
    df = df.drop_duplicates("text", keep='first')

    return df
# Followings

# def getFollowings(screen_name,followings=5,cnt=10):
#     client = get_twitter_client()
#     api = tweepy.API(auth)
#     AllTweets=[]
#     for friend in tweepy.Cursor(client.get_friends,screen_name=screen_name).items(followings):
#         tweets= client.user_timeline(screen_name=friend.screen_name,
#                                 # 200 is the maximum allowed count
#                                 count=cnt,
#                                 include_rts = False,
#                                 # Necessary to keep full_text
#                                 # otherwise only the first 140 words are extracted
#                                 tweet_mode = 'extended'
#                                 )
#         # print(friend.screen_name)
#         parsed_tweets=[]
#         parsed_tweets.append(f"{friend.screen_name}")
#         for tweet in tweets:
#             if tweet.lang == "en":
#                 filteredText = tweet.full_text.split("https")[0]

#                 parsed_tweets.append(filteredText)
#                 # print(parsed_tweets)
#         AllTweets.append(parsed_tweets)

#     return AllTweets

# For urdu Tweets
# (3). Function creating final dataframe


def get_tweets_from_urduuser(twitter_user_name, page_limit=1, count_tweet=50):
    client = get_twitter_client()

    all_tweets = []
    api = tweepy.API(auth)
    for page in Cursor(client.user_timeline,
                       screen_name=twitter_user_name,
                       count=count_tweet, tweet_mode='extended').pages(page_limit):
        for tweet in page:
            parsed_tweet = {}

            if tweet.lang == "ur":
                filteredText = tweet.full_text.split("https")[0]
                parsed_tweet['text'] = filteredText
                all_tweets.append(parsed_tweet)
    # Create dataframe
    df = pd.DataFrame(all_tweets)

    # Revome duplicates if there are any
    df = df.drop_duplicates("text", keep='first')

    return df

# Urdu Followings


def geturduFollowings(screen_name, followings=10, cnt=5):
    client = get_twitter_client()
    api = tweepy.API(auth)
    AllTweets = []
    for friend in tweepy.Cursor(client.get_friends, screen_name=screen_name).items(followings):
        tweets = client.user_timeline(screen_name=friend.screen_name,
                                      # 200 is the maximum allowed count
                                      count=cnt,
                                      include_rts=False,
                                      # Necessary to keep full_text
                                      # otherwise only the first 140 words are extracted
                                      tweet_mode='extended'
                                      )
        # print(friend.screen_name)
        parsed_tweets = []
        parsed_tweets.append(f"{friend.screen_name}")
        for tweet in tweets:
            if tweet.lang == "ur":
                filteredText = tweet.full_text.split("https")[0]

                parsed_tweets.append(filteredText)
                # print(parsed_tweets)
        AllTweets.append(parsed_tweets)

    return AllTweets

# def clean_str(string):
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = 309 - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def predict_sentence(model_path, input_data):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(
        input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['Positive', 'Negative', 'Neutral']
    results = []

    for i in range(len(classes)):
        result = {
            'class_label': classes[i],
            'score': output_data[0][i]
        }
        results.append(result)

    return results

    # neturl


# Function for Urdu Model *********************************************************************************
# cleaning strings
def pad_sentences_urdu(sentences, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = 152 - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def removing_unwanted_data(string):
    string = re.sub(r'https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE)
    string = re.sub(r'\<a href', ' ', string)
    string = re.sub(r'&amp;', '', string)
    string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', string)
    string = re.sub(r'<br />', ' ', string)
    string = re.sub(r'\'', ' ', string)

    # Tokenize each word
    string = nltk.WordPunctTokenizer().tokenize(string)

    return string


def urduuser(request):
    pos = 0
    neg = 0
    neu = 0
    tweets_array = []
    tweets_result_array = []
    context = {}
    model_path_urdu = "/home/nasir/FYP/After30%/CNNwith4/converted_model_urdu3.tflite"
    if request.method == "POST":
        if request.POST.get('usernametwitter'):
            acountname = request.POST.get('usernametwitter')
            # myTweets =  get_tweets_from_urduuser(acountname)
            # products_list = myTweets.values.tolist()

            products_list = [
                ["چیرمین عمران خان کی کال پر امپورٹڈ حکومت اور مہنگائی کے خلاف ضلع صوابی کے تحصیل ٹوپی میں احتجاج۔#الیکشن_واحد_حل "],
                ["ایک بار پھررات کے اندھیرے میں عوام پر پیٹرول بم گِرادیاگیا۔جب عوام پرزبردستی نااہل اورعوام دشمن حکومت مسلط کی جائے توپھر ملک کا یہی حال ہوگا۔امپورٹڈ حکومت کواپنی عیاشیوں سے فرصت نہیں ملتی توعوام کی طرف خاک متوجہ ہوگی۔اب وقت آگیا ہے کہ اس حکومت سے ہرحال میں عوام کی جان چھڑائی جائے"],
                ["حکومت کے فیصلوں سے میں بہت ناراض ہوں-۔ حکومت کا معیشتی حالات پر کوئی نظرانداز نہیں کیا جا سکتا۔"],
                ["ہمارا مطالبہ جمہوری ہےکہ موجودہ ناجائز اور نااہل حکومت کے پاس عوامی مینڈیٹ نہیں ہے جلد از جلد صاف شفاف انتخابات کے ذریعے اصل عوامی نمائندوں کو انتقال اقتدار کیا جائے۔بجلی کے بلوں میں اضافے کے ساتھ لوڈشیڈنگ اوربدترین مہنگائی کی وجہ سے عوام کا جینا دوبھر ہو چکا ہے۔ #الیکشن_واحد_حل "]
               
            ]

            context = process_single_user_urdu(acountname, model_path_urdu)
    return render(request, 'urduusers.html', context)


def clean_str_urdu(string):
    print("string:", string, "\n", "type string:", type(string))
    string = re.sub(r'https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE)
    string = re.sub(r'\<a href', ' ', string)
    string = re.sub(r'&amp;', '', string)
    string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', string)
    string = re.sub(r'<br />', ' ', string)
    string = re.sub(r'\'', ' ', string)
    return string


def predict_sentence_urdu(model_path_urdu, input_data):
    interpreter = tf.lite.Interpreter(model_path=model_path_urdu)
    input_details = interpreter.get_input_details()

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the expected input shape
    expected_input_shape = input_details[0]['shape']
    # Reshape the input data
    input_data = input_data[:, :expected_input_shape[1]]

    interpreter.set_tensor(
        input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['Positive', 'Negative', 'Neutral']
    results = []

    for i in range(len(classes)):
        result = {
            'class_label': classes[i],
            'score': output_data[0][i]
        }
        results.append(result)

    return results


def prepare_input_urdu(text):
    x_text = [clean_str_urdu(text)]
    x_text = [s.split(" ") for s in x_text]
    print("X text:", x_text)
    sentences_padded = pad_sentences_urdu(x_text)
    vocabulary = np.load(
        r'/home/nasir/FYP/After30%/CNNwith4/data1-vocabcls3.npy', allow_pickle=True).item()
    cleaned = []
    for word in sentences_padded:
        for word2 in word:
            if word2 in vocabulary:
                cleaned.append(word2)
            else:
                sentences_padded[0].remove(word2)
                sentences_padded[0].append("<PAD/>")
                cleaned.append("<PAD/>")
    x2 = np.array([[vocabulary.get(word2, 0) for word2 in cleaned]
                  for cleaned in sentences_padded])
    return x2.astype(np.float32)


def process_single_user_urdu(twitter_user_name, model_path):
    tweets_dict = {}
    pos = 0
    neg = 0
    neu = 0
    suggestions = 0

    # Get tweets from the user
    # myTweets = get_tweets_from_user(twitter_user_name)
    # products_list = myTweets.values.tolist()
    
    products_list = [
                ["چیرمین عمران خان کی کال پر امپورٹڈ حکومت اور مہنگائی کے خلاف ضلع صوابی کے تحصیل ٹوپی میں احتجاج۔#الیکشن_واحد_حل "],
                ["ایک بار پھررات کے اندھیرے میں عوام پر پیٹرول بم گِرادیاگیا۔جب عوام پرزبردستی نااہل اورعوام دشمن حکومت مسلط کی جائے توپھر ملک کا یہی حال ہوگا۔امپورٹڈ حکومت کواپنی عیاشیوں سے فرصت نہیں ملتی توعوام کی طرف خاک متوجہ ہوگی۔اب وقت آگیا ہے کہ اس حکومت سے ہرحال میں عوام کی جان چھڑائی جائے"],
                ["حکومت کے فیصلوں سے میں بہت ناراض ہوں-۔ حکومت کا معیشتی حالات پر کوئی نظرانداز نہیں کیا جا سکتا۔"],
                ["ہمارا مطالبہ جمہوری ہےکہ موجودہ ناجائز اور نااہل حکومت کے پاس عوامی مینڈیٹ نہیں ہے جلد از جلد صاف شفاف انتخابات کے ذریعے اصل عوامی نمائندوں کو انتقال اقتدار کیا جائے۔بجلی کے بلوں میں اضافے کے ساتھ لوڈشیڈنگ اوربدترین مہنگائی کی وجہ سے عوام کا جینا دوبھر ہو چکا ہے۔ #الیکشن_واحد_حل "]
               
            ]
    print("len product", len(products_list))
    print(products_list)
    tweet_dict = {}
    i=0
    for tweet_index, tweet in enumerate(products_list):
        print("tweet: ",tweet)
        tweet_text = tweet[0]
        print("tweet text:", tweet_text)
        sentences = sentence_tokenizer(tweet_text)
        tweet_results = {}
        print("sentences:", sentences)
        
        for sentence_index, sentence in enumerate(sentences):
            print("sentence:", sentence)
            cleaned_sentence = clean_str_urdu(sentence)
            print("cleaned sentence: ", cleaned_sentence)
            input_data = prepare_input_urdu(cleaned_sentence)
            print("input data: ", input_data)
            results = predict_sentence_urdu(model_path, input_data)
            max_result = max(results, key=lambda x: x['score'])
            class_label = max_result['class_label']
            score = max_result['score']
            tweet_results[f"sentence{sentence_index+1}"] = [
                cleaned_sentence, score, class_label]

        tweet_dict[f"tweet{tweet_index+1}"] = tweet_results

        if class_label == 'Positive':
            pos += 1
        elif class_label == 'Negative':
            neg += 1
        elif class_label == 'Neutral':
            neu += 1
        elif class_label == 'Suggestion':
            suggestions += 1

        total_tweets = pos + neg + neu + suggestions
        avg = ""
        if pos >= neg and pos >= neu and pos >= suggestions:
            avg = "Positive"
        elif neg >= pos and neg >= neu and neg >= suggestions:
            avg = "Negative"
        elif neu >= pos and neu >= neg and neu >= suggestions:
            avg = "Neutral"
        else:
            avg = "Suggestion"

        pos_avg = round((pos / total_tweets) * 100, 2)
        neg_avg = round((neg / total_tweets) * 100, 2)
        neu_avg = round((neu / total_tweets) * 100, 2)
        sug_avg = round((suggestions / total_tweets) * 100, 2)
        i+=1

    for i in range(1, 101):
        tweet_key = f"tweet{i}"
        if tweet_key in tweet_dict:
            tweet = tweet_dict[tweet_key]
            max_score = float('-inf')  # Initialize with a very low score
            predicted_label = None

            for j in range(1, len(tweet) + 1):
                sentence_key = f"sentence{j}"
                if sentence_key in tweet:
                    sentence_value = tweet[sentence_key]
                    sentence_text = sentence_value[0]
                    score = sentence_value[1]
                    class_label = sentence_value[2]

                    if score > max_score:
                        max_score = score
                        predicted_label = class_label
            tweet["predicted_label"] = predicted_label

    bert_urdu_analysis = bert_urdu_result(products_list)

    pos_tweets = 0
    neg_tweets = 0
    neu_tweets = 0

    # Iterate through the dictionary
    for tweet in bert_urdu_analysis.values():
        predicted_label = tweet['predicted_label']
        if predicted_label == 'positive':
            pos_tweets += 1
        elif predicted_label == 'negative':
            neg_tweets += 1
        elif predicted_label == 'neutral':
            neu_tweets += 1

    avgb = ""
    if pos_tweets >= neg_tweets and pos_tweets >= neu_tweets:
        avgb = "Positive"
    elif neg_tweets >= pos_tweets and neg_tweets >= neu_tweets:
        avgb = "Negative"
    else:
        avgb = "Neutral"

    total = pos_tweets + neg_tweets + neu_tweets

    pos_avgb = round((pos_tweets / total) * 100, 2)
    neg_avgb = round((neg_tweets / total) * 100, 2)
    neu_avgb = round((neu_tweets / total) * 100, 2)

    context = {
        'tweets_dict': tweet_dict,
        'bert_result': bert_urdu_analysis,

        'total_tweets': len(tweet_dict),
        'avg': avg,
        'pos_avg': pos_avg,
        'neg_avg': neg_avg,
        'neu_avg': neu_avg,
        'pos_tweets': pos,
        'neg_tweets': neg,
        'neu_tweets': neu,

        'avgb': avgb,
        'pos_avgb': pos_avgb,
        'neg_avgb': neg_avgb,
        'neu_avgb': neu_avgb,
        'pos_tweetsb': pos_tweets,
        'neg_tweetsb': neg_tweets,
        'neu_tweetsb': neu_tweets,
        'acountname': "user.name",
        'username': "twitter_user_name"
    }

    return context


def bert_urdu_result(tweets):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-

    # Load the BERT model and tokenizer from the transformers library
    vocab = np.load(
        '/home/nasir/FYP/After30%/CNNwith4/bertvocaburdu.npy', allow_pickle=True).item()
    custom_objects = {'TFBertModel': TFBertModel}
    bert_model_path = '/home/nasir/FYP/After30%/CNNwith4/bert_urdu_cls3.h5'
    model = load_model(bert_model_path, custom_objects=custom_objects)

    # Compile the model with the same optimizer and loss function used during training
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab=vocab)

    # Test tweets
    # tweets = [
#         ["حکومت کا کارروائیوں پر میں بہت خوش ہوں۔. حکومت نے معیشت کو بہتر بنانے کیلئے اقدامات اٹھائے ہیں۔" ],
#         ["حکومت کے فیصلوں سے میں بہت ناراض ہوں-۔ حکومت کا معیشتی حالات پر کوئی نظرانداز نہیں کیا جا سکتا۔"],
#         ["پاکستان کی حکومت نے بجٹ کی تشکیل کے حوالے سے بہت اچھی کارروائی کی ہے۔- حکومت کی برقراری میں کچھ توجہ کی ضرورت ہے۔"],
    # #    ]

    labels = ['negative', 'neutral', 'positive']

    tweets_dict = {}

    # Set the maximum sentence length
    max_length = 128

    # Iterate over the list of lists of tweets
    for i, tweet_list in enumerate(tweets):
        # Initialize the tweet dictionary
        tweet_dict = {}

        # Initialize dictionaries to store label counts and scores
        label_counts = {}
        label_scores = {}

        # Iterate over the tweets in the current list
        for j, tweet in enumerate(tweet_list):
            # Tokenize the tweet into sentences
            sentences = sentence_tokenizer(tweet)

            # Iterate over the sentences
            for k, sentence in enumerate(sentences):
                # Preprocess the sentence
                cleaned_sentence = clean_str_urdu(sentence)

                # Tokenize and pad the sentence
                tokens = tokenizer(cleaned_sentence, truncation=True,
                                   padding="max_length", max_length=max_length, return_tensors="tf")
                sentence_input_ids = tokens["input_ids"]
                sentence_attention_masks = tokens["attention_mask"]

                # Classify the sentence
                sentence_predictions = model.predict(
                    [sentence_input_ids, sentence_attention_masks])
                sentence_max_score = np.max(sentence_predictions)
                sentence_predicted_index = np.argmax(sentence_predictions)
                sentence_predicted_label = labels[sentence_predicted_index]

                # Update label counts and scores
                if sentence_predicted_label in label_counts:
                    label_counts[sentence_predicted_label] += 1
                    if sentence_predicted_label in label_scores and sentence_predicted_label == tweet_dict.get(f"sentence{k+1}", [None, 0, None])[2]:
                        label_scores[sentence_predicted_label] = max(
                            label_scores[sentence_predicted_label], sentence_max_score)
                    else:
                        label_scores[sentence_predicted_label] = sentence_max_score
                else:
                    label_counts[sentence_predicted_label] = 1
                    label_scores[sentence_predicted_label] = sentence_max_score

                # Add the sentence information to the tweet dictionary
                tweet_dict[f"sentence{k+1}"] = [cleaned_sentence,
                                                sentence_max_score, sentence_predicted_label]

        # Determine the predicted label for the tweet
        repeated_labels = [label for label,
                           count in label_counts.items() if count > 1]
        if repeated_labels:
            max_repeated_label = max(repeated_labels, key=lambda label: (
                label_counts[label], label_scores[label]))
            tweet_predicted_label = max_repeated_label
        else:
            tweet_predicted_label = max(label_counts, key=label_counts.get)

        # Add the tweet dictionary and predicted label to the tweets dictionary
        tweets_dict[f"tweet{i+1}"] = tweet_dict
        tweets_dict[f"tweet{i+1}"]["predicted_label"] = tweet_predicted_label

    # Print the resulting tweets dictionary
    return tweets_dict

# Urdu Followings


def urdufollowings(request):
    pos = 0
    neg = 0
    neu = 0
    following_neg_array = []
    following_neg_array = []
    following_p_array = []

    following_names_array = []
    pos_array = []
    neu_array = []
    neg_array = []
    if request.method == "POST":
        if request.POST.get('usernametwitter'):
            acountname = request.POST.get('usernametwitter')

            products_list = geturduFollowings(acountname)
            avg = []
            for j in range(0, len(products_list)):

                for i in range(0, len(products_list[j])):
                    if i == 0:
                        following_names_array.append(products_list[j][i])
                        continue
                    # print(products_list[j][i])

                    try:
                        result = predict_urdu_tweet(products_list[j][i])
                        # print("\n")
                        if result == 0:
                            pos = pos + 1
                        elif result == 1:
                            neg = neg + 1
                        else:
                            neu = neu + 1
                            time.sleep(10)
                    except:
                        pass
                if pos >= neg and pos >= neu:
                    avg.append("Positive")
                elif neg >= pos and neg >= neu:
                    avg.append("Negative")
                else:
                    avg.append("Neutral")

                total_tweets = pos + neg + neu

                pos_array.append(round((pos / total_tweets)*100, 2))
                neg_array.append(round((neg / total_tweets) * 100, 2))
                neu_array.append(round((neu / total_tweets)*100, 2))

            print(avg)
            context2 = {
                'following_result': zip(following_names_array, pos_array, neg_array, neu_array, avg)
            }

            return render(request, 'urdufollowings.html', context2)
    return render(request, 'urdufollowings.html')
