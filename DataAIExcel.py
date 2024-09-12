import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import re  # Import regex module


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

# Function to store email to session state and log it to a file
def store_email(email):
    if "user_email" not in st.session_state:
        st.session_state['user_email'] = email  # Store in session state

    # Write the email to a file for persistent storage
    with open("email_log.txt", "a") as file:
        file.write(email + "\n")

# Function to validate email format
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

# Title of the app
st.title("Data Analysis")

# Sidebar to input email before uploading file
email = st.sidebar.text_input("Enter your email")

if email:  # If the email is entered
    if is_valid_email(email):
        st.sidebar.success(f"Valid Email entered: {email}")

        # Store the email
        store_email(email)

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

            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        st.sidebar.error("Please enter a valid email address.")
else:
    st.warning("Please enter your email to proceed.")
