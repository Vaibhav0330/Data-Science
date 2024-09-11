import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import openai
#from config import OPENAI_API_KEY



# Set OpenAI API key
#openai.api_key = OPENAI_API_KEY

# Function to display basic statistics
def display_basic_statistics(df):
    st.write("**Basic Statistics**")
    stats = df.describe(include='all')
    st.dataframe(stats)

# Function to visualize data
def visualize_data(df):
    st.write("**Data Visualizations**")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Correlation Heatmap
    if st.checkbox("Correlation Heatmap"):
        st.write("**Correlation Heatmap**")
        if numeric_columns:
            corr = df[numeric_columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            st.pyplot(plt)
            plt.close()

    # Pairplot
    if st.checkbox("Pairplot"):
        st.write("**Pairplot**")
        if numeric_columns:
            sns.pairplot(df[numeric_columns])
            st.pyplot(plt)
            plt.close()

    # Histogram
    if st.checkbox("Histogram"):
        st.write("**Histogram**")
        if numeric_columns:
            column = st.selectbox("Select Column for Histogram", numeric_columns)
            if column:
                plt.figure(figsize=(8, 6))
                plt.hist(df[column], bins=20, color='blue', alpha=0.7)
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {column}")
                st.pyplot(plt)
                plt.close()

    # Boxplot
    if st.checkbox("Boxplot"):
        st.write("**Boxplot**")
        if numeric_columns:
            column = st.selectbox("Select Column for Boxplot", numeric_columns)
            if column:
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df[column], color='green')
                plt.title(f"Boxplot of {column}")
                st.pyplot(plt)
                plt.close()

    # Scatter Plot
    if st.checkbox("Scatter Plot"):
        st.write("**Scatter Plot**")
        if numeric_columns:
            x_column = st.selectbox("X-Axis", numeric_columns)
            y_column = st.selectbox("Y-Axis", numeric_columns)
            if x_column and y_column:
                plt.figure(figsize=(8, 6))
                plt.scatter(df[x_column], df[y_column], color='purple')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f"Scatter Plot of {x_column} vs {y_column}")
                st.pyplot(plt)
                plt.close()

    # Bar Plot
    if st.checkbox("Bar Plot"):
        st.write("**Bar Plot**")
        if categorical_columns:
            column = st.selectbox("Select Column for Bar Plot", categorical_columns)
            if column:
                plt.figure(figsize=(12, 8))
                bar_data = df[column].value_counts()
                plt.bar(bar_data.index, bar_data.values, color='orange')
                plt.xlabel(column)
                plt.ylabel("Count")
                plt.title(f"Bar Plot of {column}")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(plt)
                plt.close()

    # Pie Chart
    if st.checkbox("Pie Chart"):
        st.write("**Pie Chart**")
        if categorical_columns:
            column = st.selectbox("Select Column for Pie Chart", categorical_columns)
            if column:
                pie_data = df[column].value_counts()
                fig = px.pie(names=pie_data.index, values=pie_data.values, title=f"Pie Chart of {column}")
                st.plotly_chart(fig, use_container_width=True)

# Function to encode categorical variables
def encode_categorical(df):
    st.write("**Encoding Categorical Variables**")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        le = LabelEncoder()
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col])
        st.dataframe(df)
    else:
        st.info("No categorical columns found.")

# Function to get response from OpenAI based on the uploaded file
#'''def get_openai_response(query, df):
#    try:
#        # Create a summary of the dataframe to provide context
#        summary = df.describe(include='all').to_string()
#        st.write("Data Summary for Context:")
#        st.write(summary)
#
#        # Prepare the prompt with data summary and query
#        prompt = f"Based on the following data summary:\n\n{summary}\n\nAnswer the following question:\n{query}"
#
#        response = openai.ChatCompletion.create(
#            model="gpt-4-turbo",  # Ensure this model is available to you
#            messages=[
#                {"role": "user", "content": prompt}
#            ],
#            max_tokens=150
#        )
#        return response.choices[0].message['content'].strip()
#    except Exception as e:
#        st.error(f"OpenAI API Error: {e}")
#        return None

# Title of the app
st.title("Data Analysis")

# Sidebar for file upload
st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df)

        # Display basic statistics
        display_basic_statistics(df)

        # Handle missing data
        if st.sidebar.checkbox("Show Missing Data"):
            st.subheader("Missing Data")
            st.write(df.isnull().sum())

        # Visualize data
        if st.sidebar.checkbox("Visualize Data"):
            visualize_data(df)

        # Encode categorical variables
        if st.sidebar.checkbox("Encode Categorical Variables"):
            encode_categorical(df)

        # Custom Analysis
        #st.sidebar.subheader("Custom Analysis")
        #custom_query = st.sidebar.text_area("Enter a custom query")
        #if custom_query:
            #st.subheader("Custom Analysis Result")
            #response = get_openai_response(custom_query, df)
            #if response:
                #st.write(response)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a data file.")
    