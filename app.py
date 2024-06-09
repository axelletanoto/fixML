import streamlit as st
import pickle

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
        1: '18-24',
        2: '25-29',
        3: '30-34',
        4: '35-39',
        5: '40-44',
        6: '45-49',
        7: '50-54',
        8: '55-59',
        9: '60-64',
        10: '65-69',
        11: '70-74',
        12: '75-79',
        13: '80 or older'
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

    diabetes_features = [Age, Sex, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk]

    heart_features = [Age, Sex, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk]
    
    if st.button('Predict Diabetes'):
        diabetes_prediction = predict_diabetes(diabetes_features)
        st.success(f'Diabetes Prediction: {diabetes_prediction}')

    if st.button('Predict Heart Disease'):
        heart_prediction = predict_heart_disease(heart_features)
        st.success(f'Heart Disease Prediction: {heart_prediction}')

if __name__ == '__main__':
    main()
    
    
    
