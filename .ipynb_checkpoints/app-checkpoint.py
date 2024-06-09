import streamlit as st
import pickle
import numpy as np

diabetes_model = pickle.load(open('dia_mod.pkl', 'rb'))

heart_model = pickle.load(open('heart_mod.pkl', 'rb'))

def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    prediction = diabetes_model.predict(features)
    return prediction

def predict_heart_disease(features):
    features = np.array(features).reshape(1, -1)
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
    gen_hlth_mapping = {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Fair': 3, 'Poor': 4}
    diff_walk_mapping = {'No': 0, 'Yes': 1}

    diabetes_features = [Age, sex_mapping[Sex], high_bp_mapping[HighBP], high_chol_mapping[HighChol], chol_check_mapping[CholCheck], BMI, smoker_mapping[Smoker], stroke_mapping[Stroke], phys_activity_mapping[PhysActivity], fruits_mapping[Fruits], veggies_mapping[Veggies], hvy_alcohol_mapping[HvyAlcoholConsump], healthcare_mapping[AnyHealthcare], no_doc_cost_mapping[NoDocbcCost], gen_hlth_mapping[GenHlth], MentHlth, PhysHlth, diff_walk_mapping[DiffWalk]]

    heart_features = [Age, sex_mapping[Sex], high_bp_mapping[HighBP], high_chol_mapping[HighChol], chol_check_mapping[CholCheck], BMI, smoker_mapping[Smoker], stroke_mapping[Stroke], phys_activity_mapping[PhysActivity], fruits_mapping[Fruits], veggies_mapping[Veggies], hvy_alcohol_mapping[HvyAlcoholConsump], healthcare_mapping[AnyHealthcare], no_doc_cost_mapping[NoDocbcCost], gen_hlth_mapping[GenHlth], MentHlth, PhysHlth, diff_walk_mapping[DiffWalk]]
    
    if st.button('Predict Diabetes'):
        diabetes_prediction = predict_diabetes(diabetes_features)
        st.success(f'Diabetes Prediction: {diabetes_prediction}')

    if st.button('Predict Heart Disease'):
        heart_prediction = predict_heart_disease(heart_features)
        st.success(f'Heart Disease Prediction: {heart_prediction}')

if __name__ == '__main__':
    main()
    
    
    
