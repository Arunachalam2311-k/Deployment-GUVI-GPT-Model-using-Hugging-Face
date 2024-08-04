# %%writefile app.py

# Import libraries
import streamlit as st
import sqlite3
import bcrypt
import datetime
import re
import pytz
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# SQLite connection configuration

database_path = "arun.db"  # Path to your SQLite database file

# Establish the connection and create a cursor
connection = sqlite3.connect(database_path)
cursor = connection.cursor()

# Create 'user_info' table if it does not exist
cursor.execute('''CREATE TABLE IF NOT EXISTS user_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    registered_date TIMESTAMP,
                    last_login TIMESTAMP);''')

# Check if username data in the database
def username_data(username):
    cursor.execute("SELECT * FROM user_info WHERE username = ?", (username,))
    return cursor.fetchone() is not None

# Check if email data in the database
def email_data(email):
    cursor.execute("SELECT * FROM user_info WHERE email = ?", (email,))
    return cursor.fetchone() is not None

# Validate email format using regular expressions
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Create a new user in the database
def create_user(username, password, email):
    if username_data(username):
        return 'username_data'

    if email_data(email):
        return 'email_data'

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    registered_date = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

    # Insert user data into 'user_info' table
    cursor.execute(
        "INSERT INTO user_info (username, password, email, registered_date) VALUES (?, ?, ?, ?)",
        (username, hashed_password, email, registered_date)
    )
    connection.commit()
    return 'success'

# Verify user details
def to_verify_user(username, password):
    cursor.execute("SELECT password FROM user_info WHERE username = ?", (username,))
    record = cursor.fetchone()
    if record and bcrypt.checkpw(password.encode('utf-8'), record[0]):
        cursor.execute("UPDATE user_info SET last_login = ? WHERE username = ?",
                       (datetime.datetime.now(pytz.timezone('Asia/Kolkata')), username))
        connection.commit()
        return True
    return False


# Reset user password
def to_reset_password(username, new_password):
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute(
        "UPDATE user_info SET password = ? WHERE username = ?",
        (hashed_password, username)
    )
    connection.commit()

# Load the fine-tuned model and tokenizer
model_name_or_path = "/content/drive/MyDrive/fine_tuned_model"  # Update with the correct folder path
tokenizer_name_or_path = "/content/drive/MyDrive/fine_tuned_model"  # Update with the correct folder path

model, tokenizer = None, None  # Initialize model and tokenizer

try:
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
except OSError as e:
    st.error(f"Error loading model or tokenizer: {e}")
# Set the pad_token to eos_token if it's not already set
if tokenizer:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)

# Define the text generation function
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    if not model or not tokenizer:
        return ["Model or tokenizer not loaded properly."]

    # Tokenize the input text with padding
    inputs = tokenizer(seed_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences
    )

    # Decode generated text
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Streamlit app layout
st.title("GUVI GPT Model LLM PROJECT")
st.info("Disclaimer: This project is an independent initiative and has no affiliation with the GUVI company. It is solely my own project and any views or opinions expressed are not representative of GUVI.")

# Manage session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.subheader("Welcome!")
    seed_text = st.text_input("Enter seed text for GPT-2:")
    max_length = st.slider("Max length", min_value=10, max_value=500, value=100)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)

    if st.button("Generate Text"):
        if seed_text:
            generated_texts = generate_text(model, tokenizer, seed_text, max_length, temperature)
            for text in generated_texts:
                st.write(text)
        else:
            st.error("Please enter seed text.")

else:
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Login":
        st.subheader("Login Page")

        # Login form
        with st.form(key='login_form'):
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            submit_button = st.form_submit_button(label='Login')

            if submit_button:
                if to_verify_user(username, password):
                    st.success("Login successful!")
                    st.session_state.logged_in = True
                    st.rerun()  # Refresh to show the next page
                else:
                    st.error("Invalid username or password.")

    elif choice == "Sign Up":
        st.subheader("Sign Up Page")

        # Sign Up form
        with st.form(key='signup_form'):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            submit_button = st.form_submit_button(label='Sign Up')

            if submit_button:
                if not is_valid_email(email):
                    st.error("Invalid email format.")
                elif create_user(username, password, email) == 'success':
                    st.success("User created successfully!")
                    st.session_state.logged_in = True
                    st.rerun()  # Refresh to show the next page
                elif create_user(username, password, email) == 'username_data':
                    st.error("Username already exists.")
                elif create_user(username, password, email) == 'email_data':
                    st.error("Email already exists.")







# librarys

# !pip install streamlit
# pip install bcrypt
# pip install pytz
# pip install bcrypt
# !pip install pyngrok
# !npm install localtunnel
# !streamlit run /content/app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com



# "Read Data from SQLite Database into Pandas DataFrame"

# import sqlite3
# import pandas as pd

# # Connect to the SQLite database
# conn = sqlite3.connect('/content/arun.db')  # Adjust the path and filename as needed

# # Read a table into a DataFrame
# df = pd.read_sql_query("SELECT * FROM user_info", conn)

# # Close the connection
# conn.close()
# df