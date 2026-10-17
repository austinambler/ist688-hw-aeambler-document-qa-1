import streamlit as st
from openai import OpenAI, AuthenticationError

hw1 = st.Page('HW1.py', title = "HW 1")
hw2 = st.Page('HW2.py', title = 'HW 2')
hw3 = st.Page('HW3.py', title = 'HW 3')
hw4 = st.Page('HW4.py', title = 'HW 4')
hw5 = st.Page('HW5.py', title = 'HW 5')
hw7 = st.Page('HW7.py', title = 'HW 7', default= True)

pg = st.navigation([hw1, hw2, hw3, hw4, hw5, hw7])
st.set_page_config(page_title = "HW Manager")
pg.run()