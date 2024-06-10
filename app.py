import streamlit as st
import pickle
import numpy as np
import pandas as pd

diabetes_model = pickle.load(open('dia_mod.pkl', 'rb'))

heart_model = pickle.load(open('heart_mod.pkl', 'rb'))

def predict_diabetes(features):
    prediction = diabetes_model.predict(features)
    return prediction

def predict_heart_disease(features):
    prediction = heart_model.predict(features)
    return prediction

def main():
    st.title('Diabetes and Heart Disease Prediction')
    age_mapping = {
        '18-24': 1,
        '25-29': 2,
        '30-34': 3,
        '35-39': 4,
        '40-44': 5,
        '45-49': 6,
        '50-54': 7,
        '55-59': 8,
        '60-64': 9,
        '65-69': 10,
        '70-74': 11,
        '75-79': 12,
        '80 or older': 13
    }
    Age = st.radio('Age Category', list(age_mapping.keys()), format_func=lambda x: age_mapping[x])
    Sex = st.radio('Sex', ['Female', 'Male'])
    HighBP = st.radio('High BP', ['No', 'Yes'])
    HighChol = st.radio('High Cholesterol', ['No', 'Yes'])
    CholCheck = st.radio('Cholesterol in 5 years', ['No', 'Yes'])
    BMI = st.slider('BMI', 12.0, 98.0)
    Smoker = st.radio('Smoker', ['No', 'Yes'])
    Stroke = st.radio('Stroke', ['No', 'Yes'])
    PhysActivity = st.radio('Physical Activity in past 30 days', ['No', 'Yes'])
    Fruits = st.radio('Consume fruits 1 or more times per day', ['No', 'Yes'])
    Veggies = st.radio('Vegetables 1 or more times per day', ['No', 'Yes'])
    HvyAlcoholConsump = st.radio('Heavy Alcohol Consumption', ['No', 'Yes'])
    AnyHealthcare = st.radio('Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc', ['No', 'Yes'])
    NoDocbcCost = st.radio('Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?', ['No', 'Yes'])
    GenHlth = st.radio('General Health', ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
    MentHlth = st.slider('Days of poor mental health [1-30]', 0.0, 30.0)
    PhysHlth = st.slider('Physical illness in past 30 days', 0.0, 30.0)
    DiffWalk = st.radio('Do you have serious difficulty walking or climbing stairs?', ['No', 'Yes'])

    sex_mapping = {'Female': 0, 'Male': 1}
    high_bp_mapping = {'No': 0, 'Yes': 1}
    high_chol_mapping = {'No': 0, 'Yes': 1}
    chol_check_mapping = {'No': 0, 'Yes': 1}
    smoker_mapping = {'No': 0, 'Yes': 1}
    stroke_mapping = {'No': 0, 'Yes': 1}
    phys_activity_mapping = {'No': 0, 'Yes': 1}
    fruits_mapping = {'No': 0, 'Yes': 1}
    veggies_mapping = {'No': 0, 'Yes': 1}
    hvy_alcohol_mapping = {'No': 0, 'Yes': 1}
    healthcare_mapping = {'No': 0, 'Yes': 1}
    no_doc_cost_mapping = {'No': 0, 'Yes': 1}
    gen_hlth_mapping = {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
    diff_walk_mapping = {'No': 0, 'Yes': 1}

    features = np.array([
        age_mapping[Age],
        sex_mapping[Sex],
        high_bp_mapping[HighBP],
        high_chol_mapping[HighChol],
        chol_check_mapping[CholCheck],
        BMI,
        smoker_mapping[Smoker],
        stroke_mapping[Stroke],
        phys_activity_mapping[PhysActivity],
        fruits_mapping[Fruits],
        veggies_mapping[Veggies],
        hvy_alcohol_mapping[HvyAlcoholConsump],
        healthcare_mapping[AnyHealthcare],
        no_doc_cost_mapping[NoDocbcCost],
        gen_hlth_mapping[GenHlth],
        MentHlth,
        PhysHlth,
        diff_walk_mapping[DiffWalk]
    ]).reshape(1, -1)
    
    if st.button('Predict Diabetes'):
        diabetes_prediction = predict_diabetes(features)
        result = 'Diabetes Detected' if diabetes_prediction[0] == 1 else 'No Diabetes'
        st.success(f'Diabetes Prediction: {result}')

    if st.button('Predict Heart Disease'):
        heart_prediction = predict_heart_disease(features)
        result = 'Heart Disease Detected' if heart_prediction[0] == 1 else 'No Heart Disease'
        st.success(f'Heart Disease Prediction: {result}')

if __name__ == '__main__':
    main()
    
    
    
