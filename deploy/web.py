# import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import sklearn
import pickle
import copy
import re
import sys
import vncorenlp
from vncorenlp import VnCoreNLP
import nltk
from nltk import tokenize
import streamlit as st





#read lib
#stopword
f = open('stopwords.txt', 'r', encoding='UTF-8')
stopwords = f.read().split('\n')
#java library
annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')
# annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 

def NoiseDefuse(s):
    result = copy.copy(s)
    result = result.str.lower()
    result = result.apply(lambda x: re.sub(r'http\S+', '', x))
    result = result.apply(lambda x: x.replace('\n',' '))
    result = result.apply(lambda x: re.sub('[^aàảãáạăằẳẵắặâầẩẫấậ b c dđeèẻẽéẹêềểễếệ f g hiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstu ùủũúụưừửữứựvwxyỳỷỹýỵz +[0-9]+', '', x))
    return result

def reduce_dim(x):
    return x[0]

def TokenNize(s):
    return s.apply(annotator.tokenize).apply(reduce_dim)
    

def normalized1(x):
    contractions={
        'cđv': 'cổ động viên',
        'thcs': 'trung học cơ sở',
        'pgs': 'phó giáo sư ',
        'gs': 'giáo sư ',
        'ts': 'tiến sĩ ',
        'gd  đt': 'giáo dục - đào tạo',
        'gd đt': 'giáo dục - đào tạo',
        'gdđt': 'giáo dục - đào tạo',
        'hlv': 'huấn luyện viên',
        'tp': ' thành phố ',
        'hcm': ' Hồ Chí Minh ',
        'đt': 'đội tuyển',
        'gd': 'giáo dục'
    }
    for k,v in contractions.items():
        x=x.replace(k,v)
    return x

def normalized(s):
    return s.apply(normalized1)

def remove_stopword(list_word):
    clean_list = []
    for i in range(len(list_word)):
        temp=""
        temp1=""
        if i > 0 and i < (len(list_word)-1):
            temp = str(list_word[i]) + " " + str(list_word[i+1])
            temp1 = str(list_word[i-1]) + " " + str(list_word[i])
        if list_word[i] not in stopwords and temp not in stopwords and temp1 not in stopwords:
            clean_list.append(list_word[i])
    return clean_list

def Preprocess(s):
    return TokenNize(NoiseDefuse(normalized(s))).apply(remove_stopword)

def fullPreprocess(s):
    return Preprocess(s).apply(lambda x:" ".join(x))

def set_config():
    st.set_page_config(
    page_title="Fake news detection",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
         'About': "# TEAM Năm chàng lính ngự lâm"
    }
    )

def fake_news_dectection_page():
    
            
    left,mid,right = st.columns([55,1,44])
    with left:
        result = -1
        #button
        option = st.selectbox('Choose model',('Logistic Regression','Random Forest Classifier'
                                                    ,'SVC'))
        text = st.text_input('Input news')
        text_preprocessed = fullPreprocess(pd.Series([text]))
        # text_preprocessed = [text]
        if option == 'Logistic Regression':
            with open('LogisticRegression.pkl','rb') as f:
                if text != '':
                    loaded_model = pickle.load(f)
                    result = loaded_model.predict(text_preprocessed)
        elif option == 'Random Forest Classifier':
            with open('RandomForestClassifier.pkl','rb') as f:
                if text != '':
                    loaded_model = pickle.load(f)
                    result = loaded_model.predict(text_preprocessed)
                    st.write(result)
        elif option == 'SVC':
            with open('SVC.pkl','rb') as f:
                if text != '':
                    loaded_model = pickle.load(f)
                    result = loaded_model.predict(text_preprocessed)
                    
        if st.button('Predict'):
            if result != -1:
                my_bar = st.progress(0)
                for p in range(100):
                    time.sleep(0.01)    
                    my_bar.progress(p+1)
                if result == 0:
                    st.success('Real news')
                elif result == 1:
                    st.error('Fake news')
            else:
                st.write('Please input news cho predidct')
                
    with right:
        with Image.open('fake_news_wordcloud.png') as image_wc:
            st.image(image_wc, use_column_width='auto')

    
def home_page():
    
    data_member = {'ID':['19120212','19120297','19120328','19120389','19120602'],
                        'Name':['Vũ Công Duy','Đoàn Việt Nam','Võ Trọng Phú','Tô Gia Thuận','Hồ Hữu Ngọc']}
    st.subheader('ABOUT US')
    member_df = pd.DataFrame(data_member)
    st.table(member_df)


def main():
    left,mid,right = st.columns([1,1,10])
        
    with left:
        with Image.open('logo.jpg') as image_logo:
            st.image(image_logo, use_column_width='auto')
    with right:
        st.title('FAKE NEWS DETECTION')

    fake_news_dectection_page()
    if st.button('About Us'):
        home_page()
    


set_config()
main()  

    