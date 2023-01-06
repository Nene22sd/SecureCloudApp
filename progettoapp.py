import streamlit as st
import joblib
import keras
from keras.models import load_model
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



st.title('Diabetes Prediction App')
st.write('The task consists in predicting accurate blood glucose levels in Type 1 Diabetes patients. Blood glucose level prediction is a challenging task for AI researchers, with the potential to improve the health and wellbeing of people with diabetes. Knowing in advance when blood glucose is approaching unsafe levels provides time to pro-actively avoid hypo- and hyperglycaemia and their concomitant complications.')
image = Image.open('App/image.jpg')
st.image(image, use_column_width=True)
patients = st.selectbox("Number of patients",(540,544,552,559,563,567,'ModelMerge'))
uploaded_file = st.file_uploader('Choose a file',type=['xls','xlsx'])
if uploaded_file is not None:
    df1=pd.read_excel(uploaded_file)
else:
    st.warning('You need to upload excel file.')

glucose = dict()
insuline = dict()
cho = dict()

patients_sidebar = st.sidebar.selectbox("Number of patients sidebar",(540,544,552,559,563,567,'ModelMerge'))
for i in range(8):
    glucose_label = "Glucose Level"  + str(i)
    insuline_label = "Insuline Level" + str(i)
    cho_label = "Cho Level" + str(i)
    glucose[i] = st.sidebar.slider(glucose_label, 0.0, 200.00, None , 0.01)
    insuline[i] = st.sidebar.slider(insuline_label, 0.0, 3.00, None , 0.01)
    cho[i] = st.sidebar.slider(cho_label, 0.0, 12.00, None , 0.01)
button_side = st.sidebar.button('Predict side')


if (st.button('Predict')):
    string_model = 'App/model/model' + str(patients) + '.h5'
    model = load_model(string_model)
    predict_array = df1.to_numpy()
    st.write(np.shape(predict_array))
    if np.shape(predict_array) != (8,3):   
        st.warning('The excel file must contain 9 rows of 3 values ​​each')
    else:
        #st.write(df1)
        predict_array.astype(np.float32)
        #st.write(predict_array)
        predict_array_reshape = predict_array.reshape(1,24)
        array_tomodel = np.asarray([predict_array_reshape])
        result = model.predict(array_tomodel)
        st.write(result)
        """
        array_predictions = []
        for j in range(len(predict_array_reshape)):
            if j < 21: 
                array_predictions[j]=predict_array_reshape[j+3]
            else:
                array_predictions[21]=result
                array_predictions[j]=predict_array_reshape[j]
        array_tomodel = np.asarray([array_predictions])
        result2 = model.predict(array_tomodel)
        #result = model.predict(feat_cols)
        st.write(result)
        y=np.concatenate((glucose,result,result2))
        x = [0,1,2,3,4,5,6,7,13,19]
        plt.plot(x, y)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')     
        plt.title('Glucose prediction!')
        plt.show()"""
    
if button_side:
    array_prediction_side = []
    for i in range(8):
        array_prediction_side.append(glucose[i])
        array_prediction_side.append(insuline[i])
        array_prediction_side.append(cho[i])
        
    array_prediction_side_numpy = np.asarray([[array_prediction_side]])
    string_model = 'App/model/model' + str(patients_sidebar) + '.h5'
    model = load_model(string_model)
    result = model.predict(array_prediction_side_numpy)
    st.sidebar.write(result)
    




 