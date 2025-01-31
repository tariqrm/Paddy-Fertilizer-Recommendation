import streamlit as st
import psycopg2
from psycopg2 import sql
import hashlib

# PostgreSQL database configuration
DB_HOST = 'localhost'
DB_NAME = 'NewDB'
DB_USER = 'postgres'
DB_PASS = '123'

st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Welcome to the Sri Lanka Precision Agriculture Platform</h2>", unsafe_allow_html=True)

# Connect to PostgreSQL database
def get_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

# Hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Create a function for the login page
def login():
    st.title('Login')

    # Username and password input fields
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')

    # Login button
    if st.button('Login'):
        if email and password:
            try:
                conn = get_connection()
                cur = conn.cursor()
                hashed_password = hash_password(password)
                cur.execute(
                    sql.SQL("SELECT username, email FROM users WHERE email = %s AND password = %s"),
                    (email, hashed_password)
                )
                user = cur.fetchone()
                cur.close()
                conn.close()

                if user:
                    st.session_state['logged_in'] = True
                    st.session_state['email'] = user[1]
                    st.session_state['username'] = user[0]
                    st.success('Logged in successfully')
                else:
                    st.error('Invalid email or password')
            except Exception as e:
                st.error(f'Error logging in: {e}')
        else:
            st.warning('Please enter both email and password')

def logout():
    st.session_state['logged_in'] = False

# Create a function for the registration page
def register():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Join the Sri Lanka Precision Agriculture Platform</h2>", unsafe_allow_html=True)
    st.title('Register')

    # Email, username, and password input fields
    new_email = st.text_input('Email', key='register_email')
    new_password = st.text_input('Password', type='password', key='register_password')
    new_username = st.text_input('Username', key='register_username')

    # Registration button
    if st.button('Register'):
        if new_email and new_password and new_username:
            try:
                conn = get_connection()
                cur = conn.cursor()
                hashed_password = hash_password(new_password)
                cur.execute(
                    sql.SQL("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"),
                    (new_username, new_email, hashed_password)
                )
                conn.commit()
                cur.close()
                conn.close()
                st.success('Account created successfully. Please login.')
            except Exception as e:
                st.error(f'Error creating account: {e}')
        else:
            st.warning('Please enter username, email, and password')

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'email' not in st.session_state:
    st.session_state['email'] = ''

# Display the appropriate page based on the login status
if not st.session_state['logged_in']:
    st.sidebar.markdown("<h3 style='color: #4CAF50;'>Sri Lanka Precision Agriculture Platform</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: justify;'>The first app in Sri Lanka to provide precise fertilizer application recommendations. </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: justify;'>This platform provides farmers with precise fertilizer application recommendations, leveraging cutting-edge technology to optimize crop yields and reduce environmental impact.</p>", unsafe_allow_html=True)

    # Toggle between login and registration
    login_or_register = st.radio('Choose an option:', ['Login', 'Register'])

    if login_or_register == 'Login':
        login()
    elif login_or_register == 'Register':
        register()
else:
    # Main application code here
    st.write(f"You are logged in as {st.session_state['username']}.")
    # Navigation panel
    st.sidebar.markdown("<h1 style='color: #FFFFFF; background-color: #4CAF50; padding: 10px; text-align: center;'>Navigation</h1>", unsafe_allow_html=True)
    
    page = st.sidebar.radio("Go to", ["Upload your own data", "System Database", "Community", "Contact us"])
    if st.sidebar.button('Logout'):
        logout()
        st.experimental_rerun()
        
    
    if page == "Upload your own data":
        exec(open("excel.py").read())
    if page == "System Database":
        exec(open("app.py").read())
    elif page == "Community":
        exec(open("community.py").read())
    elif page == "Contact us":
        exec(open("contact.py").read())
