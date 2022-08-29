import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt

st.markdown('# Digital House <img src="https://scontent.fsdu13-1.fna.fbcdn.net/v/t39.30808-6/275930420_1550645198655229_6570428577173776959_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=f28DmQAoRYEAX-DYGY0&_nc_ht=scontent.fsdu13-1.fna&oh=00_AT_G6wfELkAt7dyckIihVED_xXU3vF_iPffZPOZy-LkJxg&oe=6312AE5D" alt="Digital House" style="display:block;margin-left:auto;margin-right:auto;width:10%">', unsafe_allow_html=True)

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