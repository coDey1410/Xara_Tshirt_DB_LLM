import streamlit as st

from langchain_helper import get_few_shot_chain



st.title("Xara T shirts : Database Query Prompt ")

question = st.text_input("Ask your Question...")

if question:
    chain=get_few_shot_chain()
    answer=chain.run(question)
    st.header("Your Answer is:")
    st.write(answer)