import pickle4 as pickle
import numpy as np
import streamlit as st

# Path ke file model
pickle_file_path = "success_percentage_model.pkl"

def welcome():
    return 'Welcome all'

def predict_success_percentage(rating, m_spend, supply_chain, sales_m, pickle_file_path):
    # Konversi input ke format array
    new_data = np.array([[float(rating), float(m_spend), float(supply_chain), float(sales_m)]])
    
    # Muat model
    with open(pickle_file_path, "rb") as file:
        model = pickle.load(file)

    # Prediksi
    predicted_success = model.predict(new_data)
    return predicted_success

def main():
    st.image("pexels-shvetsa-3962285.jpg", width=700, output_format= "auto")
    st.title("Market Success Percentage Prediction")
    st.caption("")     
    rating = st.text_input("Rating Product", "") 
    m_spend = st.text_input("Monthly Spend", "") 
    supply_chain = st.text_input("Supply Chain", "") 
    sales_m = st.text_input("Monthly Sales", "") 
    
    predicted_success = ""
    
    if st.button("Predict"):
        try:
            # Validasi input
            rating = float(rating)
            m_spend = float(m_spend)
            supply_chain = float(supply_chain)
            sales_m = float(sales_m)
            
            # Prediksi
            predicted_success = predict_success_percentage(rating, m_spend, supply_chain, sales_m, pickle_file_path)
            st.success('Success percentage of your product is {:.2f}'.format(predicted_success[0]))
            st.caption("Disclaimer: This output is based on a model trained on 500 data points in 2024, with a prediction accuracy of approximately 74% for the market. \n The machine learning model was trained using a dataset that captures product performance across various categories, focusing on sales metrics, marketing efforts, and consumer feedback.")
        except ValueError:
            st.error("Please enter numeric values for all inputs.")
    
if __name__ == '__main__':
    main()
