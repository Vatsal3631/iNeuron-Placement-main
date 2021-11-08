import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd
import numpy as np
import pickle

adaboost_model=pickle.load(open('adaboost_model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))
svm_model=pickle.load(open('svm_model.pkl','rb'))

def impute(df):
    train, test = train_test_split(df, test_size= 0.2, random_state=0)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    impute = imp_mean.fit_transform(train[['salary']])
    train.drop("salary",axis=1,inplace=True)
    train['salary'] = impute
    impute = imp_mean.transform(test[['salary']])
    test.drop("salary",axis=1,inplace=True)
    test['salary'] = impute
    df = train.append(test)
    df = df.round(0)
    df = df.apply(LabelEncoder().fit_transform)
    return df

def classify(num):
    if num == 0:
        return 'It will be challenging for you to be placed.'
    elif num == 1:
        st.balloons()
        return 'You will be placed successfully.'
    
def main():
    df=pd.read_csv("train.csv")
    df.drop("sl_no",axis=1,inplace=True)
    df = impute(df)
    
    
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Prediction of Student Placement using AdaBoost</h2>
    </div><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    algos = ['AdaBoost','Logistic','SVM']
    st.sidebar.markdown('**_Algorithm for classification_**')
    algo_option=st.sidebar.selectbox('',algos)
    st.subheader(algo_option)
    
    st.markdown('<h4><b>Your Gender</b></h4>', unsafe_allow_html=True)
    gender_radio = st.radio (label="", options=("Male","Female"))
    if gender_radio=='Male':
        gender = 0
    elif gender_radio=='Female':
        gender = 1
    
    st.markdown('<h4><b>SSC Percentage(%)</b></h4>', unsafe_allow_html=True)
    ssc_p = st.text_input(label='',value='in %',key='0')
    st.markdown('<h4><b>SSC Board</b></h4>', unsafe_allow_html=True)
    ssc_board = st.selectbox(label="",options=["Central","Others"],key='0')
    if ssc_board=='Others':
        ssc_b = 1
    elif ssc_board=='Central':
        ssc_b = 0
    
    st.markdown('<h4><b>HSC Percentage(%)</b></h4>', unsafe_allow_html=True)
    hsc_p = st.text_input(label="",value='in %',key='1')
    st.markdown('<h4><b>HSC Board</b></h4>', unsafe_allow_html=True)
    hsc_board = st.selectbox(label="",options=["Central","Others"],key='1')
    if hsc_board=='Others':
        hsc_b = 1
    elif hsc_board=='Central':
        hsc_b = 0
    
    st.markdown('<h4><b>HSC Stream</b></h4>', unsafe_allow_html=True)
    hsc_stream = st.selectbox(label='',options=["Commerce","Science","Arts"],key='2')
    if hsc_stream=='Commerce':
        hsc_s = 1
    elif hsc_stream=='Science':
        hsc_s = 2
    elif hsc_stream== 'Arts':
        hsc_s = 0
    
    st.markdown('<h4><b>Degree Percentage(%)</b></h4>', unsafe_allow_html=True)
    degree_p = st.text_input(label='',value='in %',key='2')
    st.markdown('<h4><b>Degree Type</b></h4>', unsafe_allow_html=True)
    degree_type = st.selectbox(label='',options=["Science & Technology","Commerce & Management","Others"],key='3')
    if degree_type=='Science & Technology':
        degree_t = 2
    elif degree_type=='Commerce & Management':
        degree_t = 0
    elif degree_type=='Others':
        degree_t = 1
    
    st.markdown('<h4><b>Do you have work-experience ?</b></h4>', unsafe_allow_html=True)
    work_exp = st.radio (label="", options=("Yes","No"))
    if work_exp=='Yes':
        workex = 1
    elif work_exp=='No':
        workex = 0
    
    st.markdown('<h4><b>e-Test Percentage(%)</b></h4>', unsafe_allow_html=True)
    etest_p = st.text_input(label='',value='in %',key='3')
    
    st.markdown('<h4><b>Done specialisation in</b></h4>', unsafe_allow_html=True)
    specialisation = st.radio (label="", options=("Marketing & HR","Marketing & Finance"))
    if specialisation=='Marketing & HR':
        spec = 1
    elif specialisation=='Marketing & Finance':
        spec = 0
    
    st.markdown('<h4><b>MBA Percentage(%)</b></h4>', unsafe_allow_html=True)
    mba_p = st.text_input(label='',value='in %',key='4')
    
    st.markdown('<h4><b>Package Expectation(in Lakhs)</b></h4>', unsafe_allow_html=True)
    salary = st.text_input(label='',value='in Lakh',key='5')
    
    X = df.drop('status', axis=1) 
    Y = df['status'].copy()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)
    
    st.write('\n')
    inputs=[[gender,ssc_p,ssc_b,hsc_p,hsc_p,hsc_s,degree_p,degree_t,workex,etest_p,spec,mba_p,salary]]
    if st.button('Predict'):
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete+1)
        if algo_option == 'Logistic':
            st.warning(classify(log_model.predict(inputs)))
            accuracy = log_model.score(x_test,y_test)
        elif algo_option == 'AdaBoost':
            st.warning(classify(adaboost_model.predict(inputs)))
            accuracy = adaboost_model.score(x_test,y_test)
        elif algo_option == 'SVM':
            st.warning(classify(svm_model.predict(inputs)))
            accuracy = svm_model.score(x_test,y_test)
        st.info('Model Accuracy: {:.2f} %'.format(accuracy*100))


                   
if __name__=='__main__':
                   main()
