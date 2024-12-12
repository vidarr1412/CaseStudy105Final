import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import json
from streamlit_option_menu import option_menu
# Data loading and model training function

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
def load_lottieurl(url:str):
    r =requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
st.markdown(
    """
    <style>
    .lottie-container {
        margin-left: 200px; /* Adjust the value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="lottie-container">', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
lottie_url = "https://assets9.lottiefiles.com/packages/lf20_M9p23l.json"
lottie_animation = load_lottieurl(lottie_url)

# Use columns to adjust the position


#with left_column:
    
    #st_lottie(lottie_animation, key="example")
#with right_column:
 #   st_lottie(
  #  lottie_hello,
   # speed=1,
    #reverse=False,
    #loop=True,
    #quality="low", # medium ; high

    
    #key=None,
#)
@st.cache_resource

def load_train_models2():
    # Load data
    csv_file_path = "C:/Users/Admin/Videos/Final/sealevel.csv"
    #C:\Users\Admin\Videos\Final
    try:
        data = pd.read_csv(csv_file_path)
       
       
        st.markdown("<h1 style='text-align: center;'>Sea Level Regression Model Tester and Trainer</h1>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.write(data.head())
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return
        
    # Required columns check
    required_columns = ["Year", "TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                        "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
        X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
        y = data["SmoothedGSML_GIA_sigremoved"]
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.1),
        "Ridge Regression": Ridge(alpha=0.1),
        "AdaBoost": AdaBoostRegressor(n_estimators=50),
        "Random Forest": RandomForestRegressor(n_estimators=100),
    
        "Decision Tree": DecisionTreeRegressor(),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
        "ElasticNet": ElasticNet(alpha=0.9, l1_ratio=0.9)
    }

    best_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        mae_per_fold = []
        mse_per_fold = []
        r2_per_fold = []
        
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mae_per_fold.append(mae)
            
            mse = mean_squared_error(y_test, y_pred)
            mse_per_fold.append(mse)
            
            r2 = r2_score(y_test, y_pred)
            r2_per_fold.append(r2)
        
        best_mae = min(mae_per_fold)
        best_mse = min(mse_per_fold)
        best_r2 = max(r2_per_fold)

        best_results.append({
            "Model": model_name,
            "MAE": best_mae,
            "MSE": best_mse,
            "R²": best_r2
        })

    best_results_df = pd.DataFrame(best_results)
    return best_results_df, models, X, y, kf


# Displaying the stages outside the cached function
def display_stage( best_results_df, models, X, y, kf):
    # Stage 1 - MAE Results and Boxplot
    with st.sidebar:
        selected = option_menu(
            menu_title = "Navigation",
            options=["Home", "Model Comparison","Model Selection","Hyperparameter Tuning"],
            icons=["house","bar-chart-line","capslock","thermometer-low"],
            menu_icon="cast",
            styles={
        "container": {"padding": "0!important", "background-color": "#262730"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#00000080"},
        
    }
          
        )
     
    

    

    if selected == "Home":
            # Convert the results to a DataFrame for easy display

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

        left_column, right_column = st.columns([4, 3])  # Adjust ratios as needed
        with left_column:   
            results_df = pd.DataFrame(best_results_df)
            
            # Display the table with all results
            st.subheader("Model Performance Comparison")
            st.write(results_df)

            # Identify the best model
            best_model = results_df.loc[results_df['R²'].idxmax()]
            best_model_name = best_model['Model']
            best_model_r2 = best_model['R²']
            best_model_mae = best_model['MAE']
            best_model_mse = best_model['MSE']

            # Train the best model on the entire dataset
            
            best_model_instance = models[best_model_name]
            best_model_instance.fit(X, y)
            
        # Save the trained model
        #model_filename = f"{best_model_name.replace(' ', '_')}_best_model.joblib"
        #joblib.dump(best_model_instance, model_filename)

        # Provide a download button for the saved model
        #st.subheader("Download the Best Model")
        #with open(model_filename, "rb") as model_file:
        #   st.download_button(
        #      label="Download Best Model",
        #     data=model_file,
            #    file_name=model_filename,
            #   mime="application/octet-stream"
            #)
        st.write("To be able to visualize the results , proceed to the next page")
        with right_column:
        # Conclusion text
            st.write("")  # Blank line for spacing
            
            conclusion = f"""
<p style="text-align: justify;">
Based on the results of the cross-validation, the best model for this dataset is the <b>{best_model_name}</b>.
This model achieved the highest R² value of <b>{best_model_r2:.4f}</b>, indicating it explains the most variance in the target variable.
Additionally, it performed well with the lowest Mean Absolute Error (MAE) of <b>{best_model_mae:.4f}</b> and the lowest Mean Squared Error (MSE) of <b>{best_model_mse:.4f}</b>,
making it the most suitable model for predicting the target variable on this dataset.
</p>
"""
            st.subheader("Conclusion")
            st.write("")  # Add more for increased margin
            st.markdown(conclusion, unsafe_allow_html=True)
    
            
        
    # Stage 2 - MSE Results and Bar Chart
    if selected == "Model Comparison":
            st.subheader("Comparison Diagrams")
            fig_mae_mse = go.Figure()
            fig_mae_mse.add_trace(go.Bar(
                x=best_results_df["Model"],
                y=best_results_df["MAE"],
                name="Mean MAE",
                marker_color="blue"
            ))
            fig_mae_mse.add_trace(go.Bar(
                x=best_results_df["Model"],
                y=best_results_df["MSE"],
                name="Mean MSE",
                marker_color="red"
            ))
            fig_mae_mse.update_layout(
                title='MAE and MSE for Each Model',
                xaxis_title='Model',
                yaxis_title='Score',
                template='plotly_dark',
                barmode='group'
            )
            st.plotly_chart(fig_mae_mse)

            fig_best_r2 = go.Figure()
            fig_best_r2.add_trace(go.Bar(
                x=best_results_df["Model"],
                y=best_results_df["R²"],
                name="Mean R²",
                marker_color="orange"
            ))
            fig_best_r2.update_layout(
                title='Best R² for Each Model',
                xaxis_title='Model',
                yaxis_title='Best R²',
                template='plotly_dark',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_best_r2)
            

    if selected == "Model Selection":
            # For each model, calculate the MAE per fold and plot the mean as a line graph
            for model_name, model in models.items():
                if model_name == "Random Forest":
                    # MAE per fold for Random Forest
                    mae_per_fold = []  # Reset MAE list for Random Forest
                    for fold, (train_index, test_index) in enumerate(kf.split(X)):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mae = mean_absolute_error(y_test, y_pred)
                        mae_per_fold.append(mae)

                    # Create line graph for MAE per fold
                    mean_mae_per_fold = [sum(mae_per_fold[:i+1]) / (i+1) for i in range(len(mae_per_fold))]
                    fig_mae = go.Figure()
                    fig_mae.add_trace(go.Scatter(
                        x=[f'Fold {i+1}' for i in range(len(mae_per_fold))],
                        y=mean_mae_per_fold,
                        mode='lines+markers',
                        name=f'Mean MAE for {model_name}',
                        line=dict(color='orange'),
                        marker=dict(color='orange')
                    ))
                    fig_mae.update_layout(
                        title=f'Mean MAE per Fold for {model_name}',
                        xaxis_title='Fold',
                        yaxis_title='Mean MAE',
                        template='plotly_dark',
                    )
                    st.plotly_chart(fig_mae)

                    # Create a boxplot for MAE distribution across folds
                    fig_mae_box = go.Figure()
                    fig_mae_box.add_trace(go.Box(
                        y=mae_per_fold,
                        name=f'MAE Distribution for {model_name}',
                        boxmean=True,
                        fillcolor='skyblue',
                        line=dict(color='orange'),
                        marker=dict(color='orange'),
                    ))
                    fig_mae_box.update_layout(
                        title=f'MAE Distribution for {model_name}',
                        yaxis_title='MAE',
                        template='plotly_dark',
                    )
                    st.plotly_chart(fig_mae_box)

            

        # Stage 4 - Conclusion
        # Stage 4 - Download the best model


    if selected == "Hyperparameter Tuning":
            
            
            st.subheader("Random Forest Hyperparameter Tuning")

        # Slider for n_estimators
            n_estimators = st.slider(
                'Number of Estimators (n_estimators)',
                min_value=100,
                max_value=500,
                value=100,
                step=50
            )

            # Slider for max_depth
            max_depth = st.selectbox(
                'Max Depth of Trees (max_depth)',
                options=[None, 10, 20, 30, 40, 50],
                index=0
            )

            # Slider for min_samples_split
            min_samples_split = st.slider(
                'Minimum Samples for Splitting (min_samples_split)',
                min_value=2,
                max_value=10,
                value=2,
                step=1
            )

            # Slider for min_samples_leaf
            min_samples_leaf = st.slider(
                'Minimum Samples per Leaf (min_samples_leaf)',
                min_value=1,
                max_value=4,
                value=1,
                step=1
            )

            # Slider for number of folds (K-fold cross-validation)
            n_folds = st.slider(
                'Number of Folds (K-fold Cross Validation)',
                min_value=2,
                max_value=10,
                value=5,
                step=1
            )

            # Show selected parameters
            

            # Hyperparameter grid based on user input
            param_grid = {
                'n_estimators': [n_estimators],
                'max_depth': [max_depth] if max_depth is not None else [None],
                'min_samples_split': [min_samples_split],
                'min_samples_leaf': [min_samples_leaf]
            }

            # Random Forest model
            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

            # KFold Cross Validation
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            mae_per_fold = []  # Reset MAE list for Random Forest
            mse_per_fold = []  # Reset MSE list for Random Forest
            r2_per_fold = []   # Reset R² list for Random Forest
            progress_bar = st.progress(0)
            status_text = st.empty()  # This will be used for dynamic text updates
            for fold, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                status_text.text(f"Fold {fold + 1} of {n_folds}: Training Model...")
                

                rf.fit(X_train, y_train)
                status_text.text(f"Fold {fold + 1} of {n_folds}: Predicting on Test Data...")
                y_pred = rf.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                status_text.text(f"Fold {fold + 1} of {n_folds}: Computing MAE, MSE, and R²...")
                mae_per_fold.append(mae)
                mse_per_fold.append(mse)
                r2_per_fold.append(r2)
                progress_percentage = (fold + 1) / n_folds * 100
                progress_bar.progress((fold + 1) / n_folds)  # Update progress bar
                

            

                #time.sleep(0.5)
            # Calculate the mean MAE, MSE, and R² across all folds
            mean_mae = sum(mae_per_fold) / len(mae_per_fold)
            mean_mse = sum(mse_per_fold) / len(mse_per_fold)
            mean_r2 = sum(r2_per_fold) / len(r2_per_fold)
            status_text.text(f"Model Hyperparameter Tuning Complete")
            # Plotting the Mean MAE per fold (line graph)
            plot_choice = st.radio(
                "Select Plot to Display",
            options=["MAE per Fold (Line Plot)", "MAE Distribution (Boxplot)"]
            )

            if plot_choice == "MAE per Fold (Line Plot)":
                # Plotting the Mean MAE per fold (line graph)
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    x=[f'Fold {i+1}' for i in range(len(mae_per_fold))],
                    y=mae_per_fold,
                    mode='lines+markers',
                    name=f'MAE per Fold for Random Forest',
                    line=dict(color='orange'),
                    marker=dict(color='orange')
                ))
                fig_mae.update_layout(
                    title=f'MAE per Fold for Random Forest',
                    xaxis_title='Fold',
                    yaxis_title='MAE',
                    template='plotly_dark',
                )
                st.plotly_chart(fig_mae)

            elif plot_choice == "MAE Distribution (Boxplot)":
                # Plotting the MAE distribution across all folds (boxplot)
                fig_mae_box = go.Figure()
                fig_mae_box.add_trace(go.Box(
                    y=mae_per_fold,
                    name='MAE Distribution for Random Forest',
                    boxmean=True,
                    fillcolor='skyblue',
                    line=dict(color='orange'),
                    marker=dict(color='orange'),
                ))
                fig_mae_box.update_layout(
                    title=f'MAE Distribution for Random Forest',
                    yaxis_title='MAE',
                    template='plotly_dark',
                )
                st.plotly_chart(fig_mae_box)
                df1=pd.DataFrame()
                df = pd.read_csv('C:/Users/Admin/Videos/Final/sealevel.csv')
                df1['Year'] = df['Year']
                df1['SmoothedGSML_GIA'] = df['SmoothedGSML_GIA']

                ts = df1.rename(columns={'Year':'ds', 'SmoothedGSML_GIA':'y'})

                ts.columns=['ds','y']
                model1 = Prophet( yearly_seasonality=True)
                model1.fit(ts)


                # predict for 1 year in the furure and MS - month start is the frequency
                future = model1.make_future_dataframe(periods = 60, freq='Y')  
                # now lets make the forecasts
                forecast = model1.predict(future)
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

                model1.plot(forecast)
                plt.xlabel('Year')
                plt.ylabel('Smoothed GSML w/ GIA')
                # Display the best performance metrics
                st.write(f"Performance across {n_folds}-fold cross-validation:")
            y_pred = rf.predict(X)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            st.write(f"Best Model Performance:")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"R²: {r2:.4f}")
            model_filename = "sealevel.joblib"
            joblib.dump(rf, model_filename)

            with open(model_filename, "rb") as f:
                model_data = f.read()

            st.download_button(
                label="Download Trained Model",
                data=model_data,
                file_name=model_filename,
                mime="application/octet-stream"
    )
            
    
        
            # Option to go back to Stage 4 or next stage
            

# Initialize stage if not present
  

# Call the function to load models and display stages
best_results_df, models, X, y, kf = load_train_models2()
display_stage( best_results_df, models, X, y, kf)
