import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt

st.title('Projeto integrador')

lgb_model = lgb.Booster(model_file='./LightGBMModel.txt')

fig, ax = plt.subplots(1,1)
lgb.plot_importance(lgb_model, ax=ax)

st.pyplot(fig)