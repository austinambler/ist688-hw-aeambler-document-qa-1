import streamlit as st
from openai import OpenAI, AuthenticationError

hw1 = st.Page('HW1.py', title = "HW 1")
hw2 = st.Page('HW2.py', title = 'HW 2', default = True)

pg = st.navigation([hw1, hw2])
st.set_page_config(page_title = "HW Manager")
pg.run()