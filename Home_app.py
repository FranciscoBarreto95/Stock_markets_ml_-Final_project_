import streamlit as st
from PIL import Image
title_container = st.container()
col1, col2 = st.columns([2, 18])
image = Image.open('Ironhack_logo_2.png')
with title_container:
    with col1:
        st.image(image, width=70)
    with col2:
        st.markdown('<h1 style="color: #32c3ff;">Stocks & Crypto ML prediction</h1>',
                        unsafe_allow_html=True)


#st.title('Stocks and Crypto ML tendency prediction')
st.markdown("""Creator: Francisco Barreto  
                Data Analytics - IRONHACK""")

with st.expander("Instructions on How to use this tool "):
    st.write("""
            **For Stocks- Please Select the Stock algo according to your needs:**
            - 1.1 - Stocks tendency prediction - (current day only prediction)
            - 1.2 - Stocks Price & Tendency Prophet (longterm prediction)

            **For Crypto please Please Select the Crypto algo**
            - 2 -Crypto Price & Tendency Prophet (longterm prediction)
        """)
st.sidebar.success("Select a model above.")
