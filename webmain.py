import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
import warnings
import streamlit as st
import pickle
import requests
from PIL import Image
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import time


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

from wordcloud import STOPWORDS

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
stopwords = set(STOPWORDS)
stopwords.remove("not")

# count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))
# tfidf_transformer = TfidfTransformer()
count_vect = pickle.load(open('models/cv_model.sav', 'rb'))
tfidf_transformer = pickle.load(open('models/tf_model.sav', 'rb'))


def test_sample(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    if model==svmp:
        prob = model._predict_proba_lr(sample_tfidf)[0]
    else:
        prob = model.predict_proba(sample_tfidf)[0]

    # st.write(sample.strip(), "\n %s: negative prob %0.2f, positive prob %0.2f \n\n" % (result.upper(),prob[0], prob[1]))

    return result, prob

st.set_page_config(
     page_title="Sentiment Analysis",
     page_icon="🧊",
     layout="centered",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!",
     }
 )
# nav1, nav2, nav3, nav4, nav5 = st.columns(5)

# with nav1:
#     st.button('About')
# with nav2:
#     st.button('Login')
# with nav3:
#     st.button('SignUp')
# with nav4:
#     st.button('Pricing')
# with nav5:
#     st.button('Logout')

st.header("Senitment Prediction")
st.subheader("Predict the Sentiment of reviews on any Amazon Product!")


def get_user_input():
    url = st.sidebar.text_input(label = "URL", placeholder = "Enter an Amazon product link")
    return url

HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/95.0.4638.69 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})

def getdata(url):
    r = requests.get(url, headers=HEADERS)
    return r.text

def html_code(url):
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    return (soup)

url = get_user_input() or "https://www.amazon.in/boAt-BassHeads-100-Headphones-Black/dp/B071Z8M4KX/ref=sr_1_3?crid=1DJBCZEZVQGEJ&keywords=earphones&qid=1641812109&sprefix=earph%2Caps%2C388&sr=8-3&th=1"
pid = url.split('/dp', 1)[-1][1:11]
url_half = url.split('/dp', 1)[0]

def cus_data(soup):
    data_str = ""
    cus_list = []
  
    for item in soup.find_all("span", class_="a-profile-name"):
        data_str = data_str + item.get_text()
        cus_list.append(data_str)
        data_str = ""
    return cus_list

def cus_rev(soup):

    data_str = ""
  
    for item in soup.find_all("span", class_="a-size-base review-text review-text-content"):
        data_str = data_str + item.get_text()
  
    result = data_str.split("\n")
    return (result)


page_range = st.sidebar.slider('Number of pages of reviews', min_value=1, max_value=50, value=4, help='More pages=More time')
rev_result = []
for i in range(page_range):
    url_main = url_half+'/product-reviews/'+pid+'/?reviewerType=all_reviews&pageNumber='+str(i)
    soup = html_code(url_main)
    rev_data = cus_rev(soup)
    for j in rev_data:
        if j == "":
            pass
        else:
            rev_result.append(j.strip())


st.subheader("Sentiment of Reviews: ")

MNBp = pickle.load(open('models/multinomial_nb_model.sav', 'rb'))
# prediction['Multinomial'] = MNBp.predict_proba(rev_result)[:,1]

BNBp = pickle.load(open('models/bernoulli_nb_model.sav', 'rb'))
# prediction['Bernoulli'] = BNBp.predict_proba(rev_result)[:,1]

logregp = pickle.load(open('models/logreg_model.sav', 'rb'))
# prediction['LogisticRegression'] = logregp.predict_proba(X_test_tfidf)[:,1]

random_forestp = pickle.load(open('models/random_forest_model.sav', 'rb'))

knnp = pickle.load(open('models/knn_model.sav', 'rb'))

svmp = pickle.load(open('models/svm_model.sav', 'rb'))

decision_treep = pickle.load(open('models/decision_tree_model.sav', 'rb'))




selected_model = st.sidebar.selectbox('Select a Model', ('Logistic Regression', 'Bernoulli NB', 'Multinomial NB', 'Random Forest Classifier', 'K-Nearest Neighbours', 'Support Vector Machines', 'Decision Tree Classifier'))
if selected_model=='Logistic Regression': 
    use_model = logregp
elif selected_model == 'Bernoulli NB' :
    use_model = BNBp
elif selected_model == 'Multinomial NB':
    use_model = MNBp
elif selected_model == 'Random Forest Classifier':
    use_model = random_forestp
elif selected_model == 'K-Nearest Neighbours':
    use_model = knnp
elif selected_model == 'Support Vector Machines':
    use_model = svmp
elif selected_model == 'Decision Tree Classifier':
    use_model = decision_treep
else:
    use_model = logregp

result_arr = []
p_prob_arr = []
n_prob_arr = []

if len(rev_result)!=0:
    for i in rev_result:
        # test_sample(use_model, i)
        result, prob = test_sample(use_model, i)
        result_arr.append(result.upper())
        p_prob_arr.append(prob[1])
        n_prob_arr.append(prob[0])


if result_arr.count('POS')>result_arr.count('NEG'):
    result_max = "YES"
else:
    result_max = "NO"

if(len(p_prob_arr)!=0):
    avg_pos_prob = round((sum(p_prob_arr) / len(p_prob_arr) * 100), 1)
    delta_pos = round((avg_pos_prob-50), 2)
else:
    avg_pos_prob = 0
    delta_pos = 0

if(len(n_prob_arr)!=0):
    avg_neg_prob = round((sum(n_prob_arr) / len(n_prob_arr) * 100), 1)
    delta_neg = round((avg_neg_prob-50), 2)
else:
    avg_neg_prob = 0
    delta_neg = 0

result_df = pd.DataFrame(list(zip(rev_result, result_arr, p_prob_arr, n_prob_arr)),
               columns =['Review', 'Sentiment', 'Positive Probablity', 'Negative Probablity'])
result_df['Positive Probablity'].round(decimals=2)
st.dataframe(result_df)

soups = html_code(url)
img = soups.find('img', class_='a-dynamic-image')
link = img['src']
name = img['alt']

if len(rev_result)!=0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Is this product Recommended", value=result_max)
    col2.metric("Average Positive Ratio", value=avg_pos_prob, delta=delta_pos)
    col3.metric("Average Negative Ratio", value=avg_neg_prob, delta = delta_neg)
else:
    st.warning("No Reviews Found")

st.sidebar.write('\n\n')
st.sidebar.markdown(f'## [{name}]({url})')
st.sidebar.image(link)

# components.iframe(src="https://pricehistory.in/embed/EV3F_u6", height = 360)
# components.html("""<iframe src="https://pricehistory.in/embed/EV3F_u6" name="Lavie Betula Women's Tote Handbag (Black) Totes" scrolling="no" frameborder="0" style="width: 100%; height: 360px;"></iframe>""")
# keys = prediction.keys()
# for key in ['LogisticRegression']:
#     st.write(" {}:".format(key))
#     st.write(metrics.classification_report(test["senti"], prediction.get(key)>.5, target_names = ["positive", "negative"]))
#     st.write("\n")

# for key in ['Bernoulli']:
#     st.write(" {}:".format(key))
#     st.write(metrics.classification_report(test["senti"], prediction.get(key)>.5, target_names = ["positive", "negative"]))
#     st.write("\n")

# for key in ['Multinomial']:
#     st.write(" {}:".format(key))
#     st.write(metrics.classification_report(test["senti"], prediction.get(key)>.5, target_names = ["positive", "negative"]))
#     st.write("\n")