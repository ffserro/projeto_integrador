import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt

st.write('# Digital House')
st.write('## Projeto integrador II')
st.write('### Eloá Bastos, Felipe Carvalho, Felipe Sêrro, Maiara Firmo, Pedro Aranha e Rafael Mello')
st.write('O propósito deste app é mostrar como um modelo de regressão treinado pode ser divulgado de forma que facilite o entendimento e torne os resultados mais palpáveis para o tomador de decisões.')
st.write('##')
st.write('')

lgb_model = lgb.Booster(model_file='./LightGBMModel.txt')

fig, ax = plt.subplots(1,1, figsize=(10,20))
lgb.plot_importance(lgb_model, ax=ax)

st.pyplot(fig)