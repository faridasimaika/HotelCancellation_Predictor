import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
from PIL import Image
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
#import base64
#import os
from streamlit import session_state as ss

# main page
def main():
    st.set_page_config(layout="wide")
    st.title('Hotel Booking Cancellation Prediction App')
    welcome = ''' 
          <font color='#BDD5EA' size=5>
          <center><font color='#FF3366' size=12>  </font></center>
          '''
    st.markdown(welcome, unsafe_allow_html=True)
    # sidebar



    st.sidebar.header('Input Hotel Reservation Details')
    hotel = {'City Hotel': 1,
            'Resort Hotel': 0}
    st.sidebar.selectbox('Hotel Type', options=hotel.keys(), key='hotel')
    st.sidebar.number_input('Lead Time', min_value=0, key='lead_time')
    st.sidebar.number_input("Year of Reservation", min_value=2015, max_value=2017, step=1, key='arrival_date_year')
    st.sidebar.number_input("Month of Arrival", min_value=1, max_value=12, step=1, key='arrival_date_month')
    st.sidebar.number_input("Number of Adults", min_value=0, key="adults")
    repeated_guest = {'No': 0,
                    'Yes': 1}
    st.sidebar.selectbox('Have you stayed at this hotel before?', options=repeated_guest.keys(), key='is_repeated_guest')
    st.sidebar.number_input("Number of Previous Cancellations", min_value=0, key="previous_cancellations")
    st.sidebar.number_input("How many times did you stay in the hotel?", min_value=0, key="previous_bookings_not_canceled")

    reserved_room_type = st.sidebar.subheader('Reserved Room Type')
    if reserved_room_type:
        reserved_room_type_options = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
        reserved_room_type_selected = st.sidebar.selectbox('Select Your Reserved Room Type:', reserved_room_type_options)
    else:
        reserved_room_type_selected = []

    user_inputs1 = np.zeros((1, 9))
    if 'R0' in reserved_room_type_selected:
        user_inputs1[:, 0] = 1
    if 'R1' in reserved_room_type_selected:
        user_inputs1[:, 1] = 1
    if 'R2' in reserved_room_type_selected:
        user_inputs1[:, 2] = 1
    if 'R3' in reserved_room_type_selected:
        user_inputs1[:, 3] = 1
    if 'R4' in reserved_room_type_selected:
        user_inputs1[:, 4] = 1
    if 'R5' in reserved_room_type_selected:
        user_inputs1[:, 5] = 1
    if 'R6' in reserved_room_type_selected:
        user_inputs1[:, 6] = 1
    if 'R7' in reserved_room_type_selected:
        user_inputs1[:, 7] = 1
    if 'R8' in reserved_room_type_selected:
        user_inputs1[:, 8] = 1
    #if 'R9' in reserved_room_type_selected:
        #user_inputs1[:, 9] = 1

    assigned_room_type = st.sidebar.subheader('Assigned Room Type')
    if assigned_room_type:
        assigned_room_type_options = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
        assigned_room_type_selected = st.sidebar.selectbox('Select Your Assigned Room Type:', assigned_room_type_options)
    else:
        assigned_room_type_selected = []

    user_inputs2 = np.zeros((1, 11))
    if 'A0' in assigned_room_type_selected:
        user_inputs2[:, 0] = 1
    if 'A1' in assigned_room_type_selected:
        user_inputs2[:, 1] = 1
    if 'A2' in assigned_room_type_selected:
        user_inputs2[:, 2] = 1
    if 'A3' in assigned_room_type_selected:
        user_inputs2[:, 3] = 1
    if 'A4' in assigned_room_type_selected:
        user_inputs2[:, 4] = 1
    if 'A5' in assigned_room_type_selected:
        user_inputs2[:, 5] = 1
    if 'A6' in assigned_room_type_selected:
        user_inputs2[:, 6] = 1
    if 'A7' in assigned_room_type_selected:
        user_inputs2[:, 7] = 1
    if 'A8' in assigned_room_type_selected:
        user_inputs2[:, 8] = 1
    if 'A9' in assigned_room_type_selected:
        user_inputs2[:, 9] = 1
    if 'A10' in assigned_room_type_selected:
        user_inputs2[:, 10] = 1
    #if 'A11' in assigned_room_type_selected:
        #user_inputs2[:, 11] = 1

    st.sidebar.number_input("How many times did you make changes to your booking?", min_value=0, key="booking_changes")
    st.sidebar.number_input("How many days were you wait listed?", min_value=0, key="days_in_waiting_list")
    st.sidebar.number_input("Average Daily Rate upon Reservation?", min_value=0, key="adr")
    st.sidebar.number_input("Car Parking Spaces Needed", min_value=0, key="required_car_parking_spaces")
    st.sidebar.number_input("Number of Special Requests", min_value=0, key="total_of_special_requests")
    st.sidebar.number_input("Number of Kids", min_value=0, key="kids")
    st.sidebar.number_input("Total Number of Stays", min_value=1, key="total_number_of_stays")
    st.sidebar.number_input("Number of Guests per Reservation", min_value=0, key="total_guests")
    st.sidebar.number_input("Total Number of Bookings", min_value=0, key="total_bookings")
    meal_type = st.sidebar.subheader('Meal Type')
    if meal_type:
        meal_type_options = ['Bed & Breakfast', 'Full Board', 'Half Board', 'Self Catering']
        meal_type_selected = st.sidebar.selectbox('Select Customer Meal Type:', meal_type_options)
    else:
        meal_type_selected = []

    user_inputs3 = np.zeros((1, 4))
    if 'Bed & Breakfast' in meal_type_selected:
        user_inputs3[:, 0] = 1
    if 'Full Board' in meal_type_selected:
        user_inputs3[:, 1] = 1
    if 'Half Board' in meal_type_selected:
        user_inputs3[:, 2] = 1
    if 'Self Catering' in meal_type_selected:
        user_inputs3[:, 3] = 1

    distribution_channel = st.sidebar.subheader('Distribution Channel')
    if distribution_channel:
        distribution_channel_options = ['Tour Agent', 'Direct', 'Corporate', 'Global Distribution System']
        distribution_channel_selected = st.sidebar.selectbox('Select Distribution Channel:', distribution_channel_options)
    else:
        distribution_channel_selected = []

    user_inputs4 = np.zeros((1, 4))
    if 'Tour Agent' in distribution_channel_selected:
        user_inputs4[:, 0] = 1
    if 'Direct' in distribution_channel_selected:
        user_inputs4[:, 1] = 1
    if 'Corporate' in distribution_channel_selected:
        user_inputs4[:, 2] = 1
    if 'Global Distribution System' in distribution_channel_selected:
        user_inputs4[:, 3] = 1

    international = {'Yes': 1,
                     'No': 0}
    st.sidebar.selectbox('Is the customer an international guest? (not Portuguese) ', options=international.keys(),
                         key='International')

    customer_type = st.sidebar.subheader('Customer Type')
    if customer_type:
        customer_type_options = ['Transient', 'Contract', 'Transient-Party', 'Group']
        customer_type_selected = st.sidebar.selectbox('Select Customer Type:',
                                                             customer_type_options)
    else:
        customer_type_selected = []

    user_inputs5 = np.zeros((1, 4))
    if 'Transient' in customer_type_selected:
        user_inputs5[:, 0] = 1
    if 'Contract' in customer_type_selected:
        user_inputs5[:, 1] = 1
    if 'Transient-Party' in customer_type_selected:
        user_inputs5[:, 2] = 1
    if 'Group' in customer_type_selected:
        user_inputs5[:, 3] = 1

    deposit_type = st.sidebar.subheader('Deposit Type')
    if deposit_type:
        deposit_type_options = ['No Deposit', 'Refundable', 'Non-Refundable']
        deposit_type_selected = st.sidebar.selectbox('Select Deposit Type:', deposit_type_options)
    else:
        deposit_type_selected = []

    user_inputs6 = np.zeros((1, 3))
    if 'No Deposit' in deposit_type_selected:
        user_inputs6[:, 0] = 1
    if 'Refundable' in deposit_type_selected:
        user_inputs6[:, 1] = 1
    if 'Non-Refundable' in deposit_type_selected:
        user_inputs6[:, 2] = 1
    #st.session_state["run"] = "Run"
    #st.sidebar.button("Run", key='run')
    if st.button("run"):
        instance = np.array([
            hotel[ss.hotel],
            ss.lead_time,
            ss.arrival_date_year,
            ss.arrival_date_month,
            ss.adults,
            repeated_guest[ss.is_repeated_guest],
            ss.previous_cancellations,
            ss.previous_bookings_not_canceled,
            ss.booking_changes,
            ss.days_in_waiting_list,
            ss.adr,
            ss.required_car_parking_spaces,
            ss.total_of_special_requests,
            ss.kids,
            ss.total_number_of_stays,
            ss.total_guests,
            ss.total_bookings,
            # meal_type_selected,
            user_inputs3[0][0],
            user_inputs3[0][1],
            user_inputs3[0][2],
            user_inputs3[0][3],
            # distribution_channel_selected,
            user_inputs4[0][0],
            user_inputs4[0][1],
            user_inputs4[0][2],
            user_inputs4[0][3],
            international[ss.International],
            # customer_type_selected,
            user_inputs5[0][0],
            user_inputs5[0][1],
            user_inputs5[0][2],
            user_inputs5[0][3],
            # deposit_type_selected,
            user_inputs6[0][0],
            user_inputs6[0][1],
            user_inputs6[0][2],
            #reserved_room_type,
            user_inputs1[0][0],
            user_inputs1[0][1],
            user_inputs1[0][2],
            user_inputs1[0][3],
            user_inputs1[0][4],
            user_inputs1[0][5],
            user_inputs1[0][6],
            user_inputs1[0][7],
            user_inputs1[0][8],
            #assigned_room_type_selected,
            user_inputs2[0][0],
            user_inputs2[0][1],
            user_inputs2[0][2],
            user_inputs2[0][3],
            user_inputs2[0][4],
            user_inputs2[0][5],
            user_inputs2[0][6],
            user_inputs2[0][7],
            user_inputs2[0][8],
            user_inputs2[0][9],
            user_inputs2[0][10]
        ]).reshape(1, 53)

        df = pd.read_csv("h_final.csv", index_col=0)
        df.drop("is_canceled", inplace=True, axis=1)
        scaling = StandardScaler()
        scaling.fit(df)
        instance_scaled = scaling.transform(instance)
        model = load("h_app_final.joblib")
        result = model.predict_proba(instance_scaled)
        proba = np.round(result[0][1]*100 , 2)
        st.subheader('Model Information')
        welcome = ''' 
                      <font color='#000080' size=3>
                      <p> The model used to predict the hotel cancellations is Logistic Regression with the following 
                      parameters:
                      
                      '''
        st.markdown(welcome, unsafe_allow_html=True)
        st.subheader('Model Parameters')
        col2_1, col2_2, col2_3, col2_4 = st.columns(4)

        with col2_1:
            st.info('Regularization Strength: **%s**' % 0.1)
        with col2_2:
            st.info('Solver: **%s**' % "liblinear")
        with col2_3:
            st.info('Iterations: **%s**' % 500)
        with col2_4:
            st.info('Penalty: **%s**' % "l1")

        st.subheader('Model Prediction')
        #st.write('<p style="font-size:26px; color:purple;">Model Prediction</p>',unsafe_allow_html=True)
        #st.markdown("<h1 style='font-size=60px; color: purple;'>Model Prediction</h1>", unsafe_allow_html=True)

        if proba < 50:
            st.write('<p style="font-size:25px; color:#A74482;">Probability of Cancellation: </p>',unsafe_allow_html=True)
            color = '#A74482'
        else:
            st.write('<p style="font-size:25px; color:#A74482;">Probability of Cancellation: </p>',unsafe_allow_html=True)
            color = '#693668'
        st.markdown(f'''
                <center><font color={color} size=7> {proba}% </font></center>
                ''', unsafe_allow_html=True)

        st.subheader(" Model Evaluation")
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open('roc_log.jpg')
            new_image = image.resize((1400, 1000))
            st.image(new_image, caption="ROC Curve")

        with col2:

            # original_title = '<p style="font-family:Courier; color:Black; font-size: 20px;">Confusion Matrix</p>'
            # st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)

            image = Image.open('confusion_matrix.jpg')
            new_image = image.resize((1400, 1000))
            st.image(new_image, caption="Confusion Matrix")

if __name__ == '__main__':
    main()