from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import CreateUserForm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pandas as pd
from textblob import TextBlob
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd  ### For reading the csv file
import numpy as np  ##### For joins the different array
import pickle  #### For Saving the svm model
from sklearn.feature_extraction.text import TfidfVectorizer  ### For converting numeric value
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)

                return redirect('login')

        context = {'form': form}
        return render(request, 'register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'index2.html')


count = 0
strg = ""
@login_required(login_url='login')
def predict1(request):
    url = request.POST["fulltextarea"]
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
    import numpy as np
    import nltk
    import re
    from nltk import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer  ### For converting numeric value
    import pickle

    # url = "https://www.flipkart.com/boat-storm-call-1-69-inch-hd-display-bluetooth-calling-550-nits-brightness-smartwatch/p/itmf70a25b4a16e9?pid=SMWGHYF5FXGPBV6A&lid=LSTSMWGHYF5FXGPBV6AMEEQAX&marketplace=FLIPKART&store=ajy&spotlightTagId=BestsellerId_ajy&srno=b_1_4&otracker=browse&fm=organic&iid=db45dcf7-c45f-4775-a3ba-a5acb044dd1a.SMWGHYF5FXGPBV6A.SEARCH&ppt=browse&ppn=browse&ssid=jdifw27hao0000001677015059670"
    # url="https://www.amazon.in/Samsung-Storage-MediaTek-Octa-core-Processor/dp/B0BMGB2TPR/ref=sr_1_1?pd_rd_r=94ed44e1-11b7-42d3-83a8-c076714df3de&pd_rd_w=IRhwc&pd_rd_wg=byZ4t&pf_rd_p=6e9c5ebb-d370-421b-8375-bf50155e0300&pf_rd_r=A8SYVPPXC8SJ211A1X1A&qid=1683469728&refinements=p_36%3A1318505031%2Cp_n_condition-type%3A8609960031&s=electronics&sr=1-1&th=1"

    HEADERS = ({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\AppleWebKit/537.36 (KHTML, like Gecko)\Chrome/90.0.4430.212 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'})

    req = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(req.content, 'lxml')
    # print(soup.find('div',class_="user-review-userReviewWrapper"))
    # print(soup)

    # n = url.find('?')
    # size = len(url)
    # url = url.replace("dp","product-reviews")
    # url = url[:size-n+33]

    reviews = []
    flag = 1

    # check = 1
    if "amazon" in url:
        check = 1
        print("Amazon")
        url = url.replace("/dp/", "/product-reviews/")
        url = url.replace("arp_d_product_top", "dp_d_show_all_btm")
        url += "reviewerType=all_reviews&pageNumber=0"
        print(url)
    else:
        check = 2
        url = url.replace("/p/", "/product-reviews/")
        url += "&page=1"
        print(url)

    # count = 0
    def fun(url, i):

        global count
        global flag
        if count > 7:
            flag = 0
            return

        req = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(req.content, 'lxml')
        global strg

        if check == 1:
            if (soup.find_all('div', class_="a-row a-spacing-small review-data")):
                count = 0
                print("Extracting Reviews from Page", str(i) + "...")
                for item in soup.find_all('div', class_="a-row a-spacing-small review-data"):
                    strg = strg + item.text
                    print(strg)

        else:
            if (soup.find_all('div', class_="t-ZTKy")):
                count = 0
                print("Extracting Reviews from Page", str(i) + "...")
                for item in soup.find_all('div', class_="t-ZTKy"):
                    strg += item.text
                    # print(item.get_text())
                    print(strg)
            else:
                flag = 0
                return
                # print("Retrying..")

    # n = soup.find('div',{'data-hook':"cr-filter-info-review-rating-count"})
    # n = n.text.strip().split(', '[1].split(" ")[0])
    # print(n)

    for x in range(1, 3):
        url = url[:-1]
        url += str(x)
        fun(url, x)
    strg1 = strg.replace("READ MORE", ".")
    import re

    def remove_emojis(data):
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, '', data)

    clean_text = (remove_emojis(strg1))  # no emoji

    stopWords = set(stopwords.words("english"))  ### define stopwords
    words = word_tokenize(clean_text)  ### tokanization
        # print(words)
        # Creating a frequency table to keep the
        # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()  ### converting upper to lower case
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(clean_text)  ###seperate sentence to words
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    # Average value of a sentence from the original text
    average = int(sumValues / len(sentenceValue))
    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.9 * average)):
            summary += " " + sentence

    #ab = summarize(clean_text)

    d = []
    d.append(summary)
    data = pd.read_csv("flipkart_reviews_final.csv")

    train_corpus = data['summarization']
    tf = TfidfVectorizer()
    tf.fit_transform(train_corpus)
    test_tfidf = tf.transform(d)
    loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
    test_tfidf1 = test_tfidf.toarray()
    # print(test_tfidf1)
    test_tfidf2 = np.c_[test_tfidf1]
    abc = loaded_model.predict(test_tfidf2)
    print(abc)
    context = {
    "given_review1": summary,
        "review":abc[0]
    }

    return render(request, 'result.html',context)















