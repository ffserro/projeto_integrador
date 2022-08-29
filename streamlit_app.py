import streamlit as st
import lightgbm as lgb

st.title('Projeto integrador')

lgb_model = lgb.Booster(model_file='./mode.txt')

st.pyplot(lgb_model.plot_importance)