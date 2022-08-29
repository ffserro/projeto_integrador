import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt

lgb_model = lgb.Booster(model_file='./LightGBMModel.txt')

st.markdown('# Digital House <img src="https://s3-sa-east-1.amazonaws.com/prod-jobsite-files.kenoby.com/uploads/digitalhouse-1647868655-552x368png.png" alt="Digital House" style="display:inline-block;margin-left:auto;margin-right:auto;width:10%">', unsafe_allow_html=True)

st.write('## Projeto integrador II')

st.write('O propósito deste app é mostrar como um modelo de regressão treinado pode ser divulgado de forma que facilite o entendimento e torne os resultados mais palpáveis para o tomador de decisões, contribuindo para a conexão necessária para o bom storytelling.')
st.write('##')
st.write('')

fig, ax = plt.subplots(1,1, figsize=(10,20))
lgb.plot_importance(lgb_model, ax=ax)

st.pyplot(fig)