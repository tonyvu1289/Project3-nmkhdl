import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import sklearn
import pickle



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
        
        if option == 'Logistic Regression':
            with open('LogisticRegression.pkl','rb') as f:
                if text != '':
                    loaded_model = pickle.load(f)
                    result = loaded_model.predict([text])
        elif option == 'Random Forest Classifier':
            with open('RandomForestClassifier.pkl','rb') as f:
                if text != '':
                    loaded_model = pickle.load(f)
                    result = loaded_model.predict([text])
        elif option == 'SVC':
            with open('SVC.pkl','rb') as f:
                if text != '':
                    loaded_model = pickle.load(f)
                    result = loaded_model.predict([text])
                    
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
    

if __name__ == '__main__':
    set_config()
    main()

    