import streamlit as st
import numpy as np
import pandas as pd
import io
import joblib
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json

import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie

# GitHub: https://github.com/andfanilo/streamlit-lottie
# Lottie Files: https://lottiefiles.com/

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    

st.title("Sea Level Prediction Application")

    # Load pre-trained model
uploaded_model = st.file_uploader("Upload your trained model (joblib file)", type=["joblib"])

if uploaded_model is not None:
    # Load the trained model
    model = joblib.load(uploaded_model)
    #lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
    #lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

    #st_lottie(
     #   lottie_hello,
      #  speed=1,
       # reverse=False,
       # loop=True,
       # quality="low", # medium ; high
       # renderer="svg", # canvas
       # height=None,
       # width=None,
       # key=None,
#)
    st.success("Model loaded successfully!")

    # Input section
    st.subheader("Input Data for Prediction")
   
    col1, col2 = st.columns([4, 3])

    with col1:
        year = st.date_input("Select a Year", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))

        year = year.year

   
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                   "Refer to the Year of the Observation", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

          
    with col1:
        TotalWeightedObservations = st.number_input("Total Weighted Observations", value=0.0, format="%.2f")

    
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                   "Refer to the total number of observations made in a year"
                   " Mostly more than 3,000+ observations yearly", 
                    unsafe_allow_html=True)
        
    col1, col2 = st.columns([4, 3])


    with col1:
        GMSL_noGIA = st.number_input("GMSL noGIA", value=0.0, format="%.2f")
    

    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to sea level changes in (mm) with respect to 20-year TOPEX/Jason collinear mean reference:</p>", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            
    with col1:
        StdDevGMSL_noGIA = st.number_input("Standard Deviation of GMSL noGIA", value=0.0, format="%.2f")
    

    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to the standard deviation of  the mean of the sea level changes in (mm)  </p>", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            
    with col1:
        SmoothedGSML_noGIA = st.number_input("Smoothed  GMSL noGIA", value=0.0, format="%.2f")
    

 
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to the 60-day Gaussian type filter of sea level changes in (mm) </p>", 
                    unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 3])

       
    with col1:
        GMSL_GIA = st.number_input("GMSL GIA", value=0.0, format="%.2f")
    

  
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    "Refers to the Global Mean Sea Level changes affected by the Glacial Isostatic Adjustments in (mm)</p>",                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

 
    with col1:
        StdDevGMSL_GIA = st.number_input("Standard Deviation of GMSL (GIA applied)", value=0.0, format="%.2f")
    


    with col2:
        st.markdown("<p style='font-size:12px;margin-top:25px;'>"
                    "Refers to the standard deviation of  the mean of the sea level changes affected by Glacial Isostatic Adjustments in (mm)</p>",                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

          
    with col1:
        SmoothedGSML_GIA = st.number_input("Smoothed GMSL GIA", value=0.0, format="%.2f")
    

 
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    "Refers to the 60-day Gaussian type filter of sea level changes affected by Global Isostatic Adjustments in (mm)</p>",                    unsafe_allow_html=True)
    
    
    
    input_features = np.array([[TotalWeightedObservations, GMSL_noGIA, StdDevGMSL_noGIA,
                                SmoothedGSML_noGIA, GMSL_GIA, StdDevGMSL_GIA, SmoothedGSML_GIA]])
  
    if st.button("Predict"):
        prediction = model.predict(input_features)[0]  
        st.subheader("Prediction Result")
        st.write(f"The Predicted Sea Rise Level by  {int(year):d} is  {prediction:.2f} in millimeters")


   
        
        # Save data to dataframe and provide download link
        input_data = {
            "Year": [year],
            "TotalWeightedObservations": [TotalWeightedObservations],
            "GMSL_noGIA": [GMSL_noGIA],
            "StdDevGMSL_noGIA": [StdDevGMSL_noGIA],
            "SmoothedGSML_noGIA": [SmoothedGSML_noGIA],
            "GMSL_GIA": [GMSL_GIA],
            "StdDevGMSL_GIA": [StdDevGMSL_GIA],
            "SmoothedGSML_GIA": [SmoothedGSML_GIA],
            "SmoothedGSML_GIA_sigremoved": [prediction],
            "Prediction":"Using SeaScope"
        }

        df = pd.DataFrame(input_data)

        # File path to your existing CSV file
        existing_file = 'sealevel.csv'

        try:
            # Try to load the existing CSV
            existing_df = pd.read_csv(existing_file)
            # Append the new prediction to the existing data
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            # If the file doesn't exist, create a new one
            updated_df = df

        # Save the updated DataFrame back to CSV
        updated_df.to_csv(existing_file, index=False)
        csv_file_path = "C:/Users/Admin/Videos/Final/sealevel.csv"

        try:
            data = pd.read_csv(csv_file_path)
            
            # App Title
            # Dataset preview
            

        except FileNotFoundError:
            st.error("File not found. Please check the file path.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

        # Check for required columns
        required_columns = [
            "Year", "TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"
        ]

        if all(col in data.columns for col in required_columns):
            # Extracting necessary data
            years = data["Year"]
            smoothed_gmsl_sigremoved = data["SmoothedGSML_GIA_sigremoved"]

            # Plotting the line graph using Plotly
            st.subheader("Global Sea Level Change Over Years")
            fig = go.Figure()

            # Add line trace
            fig.add_trace(go.Scatter(x=years, y=smoothed_gmsl_sigremoved, mode='lines', name="Global Sea Level Change", line=dict(color="blue")))

            # Update layout with title and labels
            fig.update_layout(
                title="Sea Level Time Series",
                xaxis_title="Year",
                yaxis_title="Global Sea Level Change",
                showlegend=True,
                width=1000 ,
                legend=dict(
        x=0,  # Position the legend inside the graph (x: 0 to 1, where 0 is left and 1 is right)
        y=0.9,  # Position the legend inside the graph (y: 0 to 1, where 0 is bottom and 1 is top)
        traceorder='normal',
        bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: make the background of the legend slightly transparent
        bordercolor='Black',
        borderwidth=1) # Increase the width of the figure
            )
            
            # Display the plot in Streamlit
            st.plotly_chart(fig)
        else:
            st.error("The required columns are not present in the dataset. Please check the dataset structure.")


        # Provide a download link for the updated CSV
        st.download_button(
            label="Download Updated Prediction Data as CSV",
            data=updated_df.to_csv(index=False),
            file_name="seaelvel.csv",
            mime="text/csv"
        )