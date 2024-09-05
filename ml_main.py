import streamlit as st 
import sklearn
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True


if os.path.exists('./s_data.csv'):
    df = pd.read_csv('s_data.csv',index_col= None)
with st.sidebar:
    st.title("ML regrssion and classification")
    choice = st.radio("Navigaton",["Upload","Regression","Classification","Download"])
    st.title("Info :")
    st.info("Explore the world of machine learning with our comprehensive platform that provides detailed insights into both regression and classification algorithms. Whether you're a beginner or an expert, our resources help you understand how these algorithms work, how they can be applied to different datasets, and how to choose the best one for your specific needs.")

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file =  st.file_uploader("Upload Your Dataset Here")
    if file:
        df= pd.read_csv(file,index_col=None)
        df.to_csv('s_data.csv',index=None)
        st.dataframe(df)

        

if choice == "Regression":
       
        st.write("""<h2 style='font-size: 24px;'>Step 1 (Preprocessing) : Handling Missing Data, Feature Scaling  """,unsafe_allow_html=True)
        # Preprocessing: Handling Missing Data
        # Dropdown for interpolation method selection
        interpolation_method = st.selectbox(
            "Select the method to fill missing values",
            ['None', 'previous', 'next', 'nearest', 'linear', 'spline', 'pchip'],label_visibility="hidden"
        )

        if interpolation_method == 'None':
            st.error("No interpolation method selected. Please select a method to fill missing values.")
        else:
            if interpolation_method == 'previous':
                df = df.ffill()
            elif interpolation_method == 'next':
                df = df.bfill()
            elif interpolation_method in ['nearest', 'linear']:
                df = df.interpolate(method=interpolation_method)
            elif interpolation_method in ['spline', 'pchip']:
                for column in df.columns:
                    if df[column].dtype in [float, int]:
                    # Apply spline or pchip interpolation
                        if df[column].isna().sum() > 0:
                            if interpolation_method == 'spline':
                                df[column] = df[column].interpolate(method='spline', order=2)
                            elif interpolation_method == 'pchip':
                                df[column] = df[column].interpolate(method='pchip', order=2)
            st.success(f"Missing values filled using {interpolation_method} method.")
           

    # Slider for the random_state parameter
        st.write("""<h2 style='font-size: 24px;'>Step 2 : Select the random_state parameter in the train_test_split function is used to control the randomness during the process of splitting a dataset into training and testing sets. Specifically, it serves the following purposes:""",unsafe_allow_html=True)
        st.write("1. Reproducibility")
        st.write("2. Consistency Across Runs")
        st.write("3. Random Seed")
        random_state1_input = st.text_input("Enter Random State for Train/Test Split",  value=42)
         # Step 1: Select Features and Target
        st.write("""<h2 style='font-size: 24px;'>Select Features and Target:""",unsafe_allow_html=True)
        
        
        
    
        all_columns = df.columns.tolist()

        # Select the target column
        target_column = st.selectbox("Select the Target Column (y)", all_columns,label_visibility="hidden")

         # Select feature columns
        feature_columns = st.multiselect("Select Feature Columns (X)", all_columns, default=[col for col in all_columns if col != target_column],label_visibility="hidden")

        if target_column and feature_columns:
            
            st.button("Split The Data into Training and Test Data",  on_click=click_button, key="split_data_button")
            if st.session_state.clicked:
                
            
                if random_state1_input.strip():
                    random_state1 = int(random_state1_input)
                else:
                    random_state1 = None  
                # Split the data into training and testing sets
                X = df[feature_columns]
                Y = df[target_column]
                x_train, x_test, y_train, y_test = train_test_split(X,Y , test_size=0.2,random_state=random_state1)
            
                # Save the datasets
                x_train.to_csv('x_train_data.csv', index=False)
                y_train.to_csv('y_train_data.csv', index=False)
                x_test.to_csv('x_test_data.csv', index=False)
                y_test.to_csv('y_test_data.csv', index=False)
            
                st.success("Data has been split into training and testing sets successfully!")
                col1, col2 = st.columns(2)
                st.write("Training Data Preview:")
                with col1:
                    st.write("x_train:")
                    st.dataframe(x_train)
                with col2:
                    st.write("y_train:")
                    st.dataframe(y_train)

                col1, col2 = st.columns(2)
                st.write("Testing Data Preview:")
                with col1:
                    st.write("x_test:")
                    st.dataframe(x_test)
                with col2:
                    st.write("y_test:")
                    st.dataframe(y_test)
                
                
        st.markdown("""<h2 style='font-size: 24px;'>Step 3: Choose a Regression Model, or Compare All Models to determine which one performs best.</h2>""",unsafe_allow_html=True)
            # models
        choice = st.radio("", ["k-Nearest Neighbors (k-NN)", "Linear Regression", "Support Vector Machines (SVM)", "Neural Networks", "Random Forests", "Compare All Models"])
        st.button("Run",  on_click=click_button, key="run_button")
        if st.session_state.clicked:
            if choice == "k-Nearest Neighbors (k-NN)":
                n_neighbors1_input = st.text_input("Enter ",  value=3)
                st.button("Number of Neighbors (k)",  on_click=click_button, key="Neighbors_button" )
                if st.session_state.clicked:
                    if n_neighbors1_input.strip():
                        n_neighbors1 = int(n_neighbors1_input)
                    else:
                        n_neighbors1 = None  
                    reg = KNeighborsRegressor(n_neighbors=n_neighbors1)
                    reg.fit(x_train, y_train)
                    y_predict = reg.predict(x_test)
                    col1, col2,col3 = st.columns(3)
                    with col1:
                        st.write("Test set predictions:" )
                        st.dataframe(y_predict.round(decimals=2))

                    with col2:
                        st.write("Ground Truth        :")
                        st.dataframe(y_test.round(decimals=2))
                    with col3:
                        st.write("Error               :")
                        st.dataframe( (y_predict-y_test).round(decimals=2))
                    Abs_error = np.sum(np.abs((y_predict-y_test))).round(decimals=2)
                    st.write("Absolute Error      :", Abs_error )
                    st.write("Test set R2-Score: {:.2f}".format(reg.score(x_test, y_test)))
                    x_min, x_max = x_train.min() - 1, x_train.max() + 1
                    
                    line = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
                    prediction = reg.predict(line)

                    # for graph
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    
                    ax.plot(x_train, y_train, '^', c=mglearn.cm2(0), markersize=8, label='Training data/target')
                    ax.plot(x_test, y_test, 'v', c=mglearn.cm2(1), markersize=8, label='Test data/target')
                    ax.plot(line, prediction, label='Model predictions')
                    ax.set_title(
                        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                        n_neighbors1, reg.score(x_train, y_train),reg.score(x_test, y_test)))
                    ax.set_xlabel("Feature")
                    ax.set_ylabel("Target")
                    st.pyplot(fig)
                    


if choice == "Classification":
    pass
if choice == "Download":
    pass