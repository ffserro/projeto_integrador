import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt

lgb_model = lgb.Booster(model_file='./LightGBMModel.txt')

st.markdown('<img src="https://s3-sa-east-1.amazonaws.com/prod-jobsite-files.kenoby.com/uploads/digitalhouse-1647868655-552x368png.png" alt="Digital House" style="display:block;margin-left:auto;margin-right:auto;margin-top:-180px;margin-bottom:-180px;width:100%">', unsafe_allow_html=True)

st.write('## Projeto integrador II')

st.write('O propósito deste app é mostrar como um modelo de regressão treinado pode ser divulgado de forma que facilite o entendimento e torne os resultados mais palpáveis para o tomador de decisões, contribuindo para a conexão necessária para o bom storytelling.')
st.write('##')
st.write('')

with st.form('enem_survey'):
    idade = st.selectbox('Quantos anos você tem?', ['-', 'Menor de 17 anos','17 anos','18 anos','19 anos','20 anos','21 anos','22 anos','23 anos','24 anos','25 anos','Entre 26 e 30 anos','Entre 31 e 35 anos','Entre 36 e 40 anos', 'Entre 41 e 45 anos', 'Entre 46 e 50 anos', 'Entre 51 e 55 anos', 'Entre 56 e 60 anos', 'Entre 61 e 65 anos', 'Entre 66 e 70 anos', 'Maior de 70 anos'])
    idade = {'-':'-',
        'Menor de 17 anos': 1,
        '17 anos': 2,
        '18 anos': 3,
        '19 anos': 4,
        '20 anos': 5,
        '21 anos': 6,
        '22 anos': 7,
        '23 anos': 8,
        '24 anos': 9,
        '25 anos': 10,
        'Entre 26 e 30 anos': 11,
        'Entre 31 e 35 anos': 12,
        'Entre 36 e 40 anos': 13,
        'Entre 41 e 45 anos': 14,
        'Entre 46 e 50 anos': 15,
        'Entre 51 e 55 anos': 16,
        'Entre 56 e 60 anos': 17,
        'Entre 61 e 65 anos': 18,
        'Entre 66 e 70 anos': 19,
        'Maior de 70 anos': 20}[idade]

    
fig, ax = plt.subplots(1,1, figsize=(10,20))
lgb.plot_importance(lgb_model, ax=ax)

st.pyplot(fig)