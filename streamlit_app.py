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

    ano_conclusao = st.selectbox('Em que ano você concluiu o Ensino Médio?', ['-'] + list(range(2019,2007,-1)) + ['Antes de 2008'])
    ano_conclusao = {j:i for i,j in enumerate(['-'] + list(range(2019,2007,-1)) + ['Antes de 2008'])}[ano_conclusao]

    renda_familiar = st.selectbox('Qual é a renda mensal da sua família (soma de todos)', ['Nenhuma renda.', 'Até R$ 998,00.', 'De R$ 998,01 até R$ 1.497,00.', 'De R$ 1.497,01 até R$ 1.996,00.', 'De R$ 1.996,01 até R$ 2.495,00.', 'De R$ 2.495,01 até R$ 2.994,00.', 'De R$ 2.994,01 até R$ 3.992,00.', 'De R$ 3.992,01 até R$ 4.990,00.', 'De R$ 4.990,01 até R$ 5.988,00.', 'De R$ 5.988,01 até R$ 6.986,00.', 'De R$ 6.986,01 até R$ 7.984,00.', 'De R$ 7.984,01 até R$ 8.982,00.', 'De R$ 8.982,01 até R$ 9.980,00.', 'De R$ 9.980,01 até R$ 11.976,00.', 'De R$ 11.976,01 até R$ 14.970,00.', 'De R$ 14.970,01 até R$ 19.960,00.', 'Mais de R$ 19.960,00.'])
    renda_familiar = {j:i+1 for i, j in enumerate(['Nenhuma renda.', 'Até R$ 998,00.', 'De R$ 998,01 até R$ 1.497,00.', 'De R$ 1.497,01 até R$ 1.996,00.', 'De R$ 1.996,01 até R$ 2.495,00.', 'De R$ 2.495,01 até R$ 2.994,00.', 'De R$ 2.994,01 até R$ 3.992,00.', 'De R$ 3.992,01 até R$ 4.990,00.', 'De R$ 4.990,01 até R$ 5.988,00.', 'De R$ 5.988,01 até R$ 6.986,00.', 'De R$ 6.986,01 até R$ 7.984,00.', 'De R$ 7.984,01 até R$ 8.982,00.', 'De R$ 8.982,01 até R$ 9.980,00.', 'De R$ 9.980,01 até R$ 11.976,00.', 'De R$ 11.976,01 até R$ 14.970,00.', 'De R$ 14.970,01 até R$ 19.960,00.', 'Mais de R$ 19.960,00.'])}[renda_familiar]

    pessoas_cohab = st.selectbox('Quantas pessoas, contando com você, atualmente moram na sua casa?', list(range(1,21)))

    cor_raca = st.selectbox('Com que cor/raça você se identifica?', ['Não declarado', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indígena'])
    cor_raca = {j:i+1 for i,j in enumerate(['Não declarado', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indígena'])}[cor_raca]

    escolaridade_mae = st.selectbox('Até que série sua mãe (ou a mulher responsável por você) estudou?', ['Nunca estudou.', 'Não completou a 4ª série/5º ano do Ensino Fundamental.', 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental.', 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio.', 'Completou o Ensino Médio, mas não completou a Faculdade.', 'Completou a Faculdade, mas não completou a Pós-graduação.', 'Completou a Pós-graduação.', 'Não sei.'])
    escolaridade_mae = {j:i+1 for i, j in enumerate(['Nunca estudou.', 'Não completou a 4ª série/5º ano do Ensino Fundamental.', 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental.', 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio.', 'Completou o Ensino Médio, mas não completou a Faculdade.', 'Completou a Faculdade, mas não completou a Pós-graduação.', 'Completou a Pós-graduação.', 'Não sei.'])}[escolaridade_mae]

    ocupacao_pai = st.selectbox('A partir da apresentação de algumas ocupações divididas em grupos ordenados, indique o grupo que contempla a ocupação mais próxima da ocupação do seu pai ou do homem responsável por você. (Se ele não estiver trabalhando, escolha uma ocupação pensando no último trabalho dele).', 
        ['Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultor, pescador, lenhador, seringueiro, extrativista.',
        'Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria.',
        'Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista.',
        'Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes etc.), técnico (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria.',
        'Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados.',
        'Não sei.'])
    ocupacao_pai = {j:i+1 for i,j in enumerate(['Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultor, pescador, lenhador, seringueiro, extrativista.',
        'Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria.',
        'Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista.',
        'Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes etc.), técnico (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria.',
        'Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados.',
        'Não sei.'])}[ocupacao_pai]

    escolaridade_pai = st.selectbox('Até que série seu pai, ou o homem responsável por você, estudou?', ['Nunca estudou.',
        'Não completou a 4ª série/5º ano do Ensino Fundamental.',
        'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental.',
        'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio.',
        'Completou o Ensino Médio, mas não completou a Faculdade.',
        'Completou a Faculdade, mas não completou a Pós-graduação.',
        'Completou a Pós-graduação.',
        'Não sei.'])
    escolaridade_pai = {j:i+1 for i,j in enumerate(['Nunca estudou.',
        'Não completou a 4ª série/5º ano do Ensino Fundamental.',
        'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental.',
        'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio.',
        'Completou o Ensino Médio, mas não completou a Faculdade.',
        'Completou a Faculdade, mas não completou a Pós-graduação.',
        'Completou a Pós-graduação.',
        'Não sei.'])}[escolaridade_pai]

    num_computador = st.selectbox('Na sua residência tem computador?', ['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_computador = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}[num_computador]

    num_televisao = st.selectbox('Na sua residência tem televisão em cores?', ['Não.',
        'Sim, uma.',
        'Sim, duas.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_televisao = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, uma.',
        'Sim, duas.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}[num_televisao]

    ocupacao_mae = st.selectbox('A partir da apresentação de algumas ocupações divididas em grupos ordenados, indique o grupo que contempla a ocupação mais próxima da ocupação da sua mãe ou da mulher responsável por você. (Se ela não estiver trabalhando, escolha uma ocupação pensando no último trabalho dela).',
        ['Grupo 1: Lavradora, agricultora sem empregados, bóia fria, criadora de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultora, pescadora, lenhadora, seringueira, extrativista.',
        'Grupo 2: Diarista, empregada doméstica, cuidadora de idosos, babá, cozinheira (em casas particulares), motorista particular, jardineira, faxineira de empresas e prédios, vigilante, porteira, carteira, office-boy, vendedora, caixa, atendente de loja, auxiliar administrativa, recepcionista, servente de pedreiro, repositora de mercadoria.',
        'Grupo 3: Padeira, cozinheira industrial ou em restaurantes, sapateira, costureira, joalheira, torneira mecânica, operadora de máquinas, soldadora, operária de fábrica, trabalhadora da mineração, pedreira, pintora, eletricista, encanadora, motorista, caminhoneira, taxista.',
        'Grupo 4: Professora (de ensino fundamental ou médio, idioma, música, artes etc.), técnica (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretora de imóveis, supervisora, gerente, mestre de obras, pastora, microempresária (proprietária de empresa com menos de 10 empregados), pequena comerciante, pequena proprietária de terras, trabalhadora autônoma ou por conta própria.',
        'Grupo 5: Médica, engenheira, dentista, psicóloga, economista, advogada, juíza, promotora, defensora, delegada, tenente, capitã, coronel, professora universitária, diretora em empresas públicas ou privadas, política, proprietária de empresas com mais de 10 empregados.',
        'Não sei.'])
    ocupacao_mae = {j:i+1 for i,j, in enumerate(['Grupo 1: Lavradora, agricultora sem empregados, bóia fria, criadora de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultora, pescadora, lenhadora, seringueira, extrativista.',
        'Grupo 2: Diarista, empregada doméstica, cuidadora de idosos, babá, cozinheira (em casas particulares), motorista particular, jardineira, faxineira de empresas e prédios, vigilante, porteira, carteira, office-boy, vendedora, caixa, atendente de loja, auxiliar administrativa, recepcionista, servente de pedreiro, repositora de mercadoria.',
        'Grupo 3: Padeira, cozinheira industrial ou em restaurantes, sapateira, costureira, joalheira, torneira mecânica, operadora de máquinas, soldadora, operária de fábrica, trabalhadora da mineração, pedreira, pintora, eletricista, encanadora, motorista, caminhoneira, taxista.',
        'Grupo 4: Professora (de ensino fundamental ou médio, idioma, música, artes etc.), técnica (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretora de imóveis, supervisora, gerente, mestre de obras, pastora, microempresária (proprietária de empresa com menos de 10 empregados), pequena comerciante, pequena proprietária de terras, trabalhadora autônoma ou por conta própria.',
        'Grupo 5: Médica, engenheira, dentista, psicóloga, economista, advogada, juíza, promotora, defensora, delegada, tenente, capitã, coronel, professora universitária, diretora em empresas públicas ou privadas, política, proprietária de empresas com mais de 10 empregados.',
        'Não sei.'])}[ocupacao_mae]

    num_celular = st.selectbox('Na sua residência tem telefone celular?',
        ['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_celular = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}[num_celular]

    tipo_escola = st.selectbox('Tipo de escola do Ensino Médio', [
'Pública'
'Privada'
'Exterior'])
    tipo_escola = {j:i+2 for i,j in enumerate([
'Pública'
'Privada'
'Exterior'])}[tipo_escola]

    tipo_lingua = st.selectbox('Qual foi a língua estrangeira escolhida para a prova?', ['Inglês', 'Espanhol'])
    tipo_lingua = {j:i for i,j in enumerate(['Inglês', 'Espanhol'])}[tipo_lingua]

    num_banheiros = st.selectbox('Na sua casa tem banheiros?', ['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_banheiros = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}[num_banheiros]

    nacionalidade = st.selectbox('Qual é a sua nacionalidade?', ['Brasileiro(a)',
        'Brasileiro(a) Naturalizado(a)',
        'Estrangeiro(a)',
        'Brasileiro(a) Nato(a), nascido(a) no exterior'])
    nacionalidade = {j:i+1 for i,j in enumerate(['Brasileiro(a)',
        'Brasileiro(a) Naturalizado(a)',
        'Estrangeiro(a)',
        'Brasileiro(a) Nato(a), nascido(a) no exterior'])}[nacionalidade]

    sexo = st.selectbox('Qual é o seu sexo?', ['Masculino', 'Feminino'])
    sexo = {'Masculino':0, 'Feminino':0}[sexo]

    num_freezer = st.selectbox('Na sua residência tem freezer (independente ou segunda porta da geladeira)?', ['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_freezer = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}[num_freezer]

    empregada_dom = st.selectbox('Na sua casa trabalha empregada doméstica?', ['Não.',
        'Sim, um ou dois dias por semana.',
        'Sim, três ou quatro dias por semana.',
        'Sim, pelo menos cinco dias por semana.'])
    empregada_dom = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um ou dois dias por semana.',
        'Sim, três ou quatro dias por semana.',
        'Sim, pelo menos cinco dias por semana.'])}[empregada_dom]

    sit_conclusao = st.selectbox('Sobre a sua conclusão do ensino médio:', ['Já concluí o Ensino Médio',
        'Estou cursando e concluirei o Ensino Médio no ano corrente',
        'Estou cursando e concluirei o Ensino Médio após o ano corrente',
        'Não concluí e não estou cursando o Ensino Médio'])
    sit_conclusao = {j:i+1 for i,j in enumerate(['Já concluí o Ensino Médio',
        'Estou cursando e concluirei o Ensino Médio no ano corrente',
        'Estou cursando e concluirei o Ensino Médio após o ano corrente',
        'Não concluí e não estou cursando o Ensino Médio'])}

    num_carro = st.selectbox('Na sua casa tem carro?', ['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_carro = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}

    estado_civil = st.selectbox('Qual é o seu estado civil?', [
        'Solteiro(a)',
        'Casado(a)/Mora com companheiro(a)',
        'Divorciado(a)/Desquitado(a)/Separado(a)',
        'Viúvo(a)'])
    estado_civil = {j:i+1 for i,j in enumerate([
        'Solteiro(a)',
        'Casado(a)/Mora com companheiro(a)',
        'Divorciado(a)/Desquitado(a)/Separado(a)',
        'Viúvo(a)'])}[estado_civil]

    num_quartos = st.selectbox('Na sua casa tem quartos para dormir?', ['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])
    num_quartos = {j:i+1 for i,j in enumerate(['Não.',
        'Sim, um.',
        'Sim, dois.',
        'Sim, três.',
        'Sim, quatro ou mais.'])}[num_quartos]
    
    num_maqlavar = st.selectbox('Na sua residência tem máquina de lavar roupa? (o tanquinho NÃO deve ser considerado)', ['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])
    num_maqlavar = {j:i+1 for i,j in enumerate(['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])}[num_maqlavar]

    tv_assinatura = st.selectbox('Na sua residência tem TV por assinatura?', ['Não.', 'Sim.'])
    tv_assinatura = {'Não.':1, 'Sim.':2}[tv_assinatura]

    dvd = st.selectbox('Na sua residência tem aparelho de DVD?', ['Não.', 'Sim.'])
    dvd = {1:'Não.', 2:'Sim.'}[dvd]

    aspirador = st.selectbox('Na sua residência tem aspirador de pó?', ['Não.', 'Sim.'])
    aspirador = {1:'Não.', 2:'Sim.'}[aspirador]
    
    tel_fixo = st.selectbox('Na sua residência tem telefone fixo?', ['Não.', 'Sim.'])
    tel_fixo = {1:'Não.', 2:'Sim.'}[tel_fixo]

    num_maqsecar = st.selectbox('Na sua residência tem máquina secar roupas? (independente ou em conjunto com a máquina de lavar)', ['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])
    num_maqsecar = {j:i+1 for i,j in enumerate(['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])}[num_maqsecar]

    num_microondas = st.selectbox('Na sua residência tem forno de microondas?', ['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])
    num_microondas = {j:i+1 for i,j in enumerate(['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])}[num_microondas]

    num_motocicleta = st.selectbox('Na sua residência tem motocicleta?', ['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])
    num_motocicleta = {j:i+1 for i,j in enumerate(['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])}[num_motocicleta]

    num_geladeira = st.selectbox('Na sua residência tem geladeira?', ['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])
    num_geladeira = {j:i+1 for i,j in enumerate(['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])}[num_geladeira]

    num_lavalouca = st.selectbox('Na sua residência tem máquina de lavar louças?', ['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])
    num_lavalouca = {j:i+1 for i,j in enumerate(['Não.'
        'Sim, uma.'
        'Sim, duas.'
        'Sim, três.'
        'Sim, quatro ou mais.'])}[num_lavalouca]

    treineiro = st.selectbox('Você fez a prova apenas para testar seus conhecimentos?', ['Não.', 'Sim.'])
    treineiro = {'Sim.':1, 'Não.':0}[treineiro]

    acesso_internet = st.selectbox('Sua residência tem acesso à internet?', ['Sim.', 'Não.'])
    acesso_internet = {'Sim.':2, 'Não.':1}[acesso_internet]

    respostas = { 
        'TP_FAIXA_ETARIA':idade,
        'TP_SEXO':sexo,
        'TP_ESTADO_CIVIL': estado_civil,
        'TP_COR_RACA': cor_raca,
        'TP_NACIONALIDADE':nacionalidade,
        'TP_ST_CONCLUSAO': sit_conclusao,
        'TP_ANO_CONCLUIU': ano_conclusao,
        'TP_ESCOLA': tipo_escola,
        'IN_TREINEIRO': treineiro,
        'TP_LINGUA': tipo_lingua, 
        'Q001': escolaridade_pai,
        'Q002': escolaridade_mae,
        'Q003':	ocupacao_pai,
        'Q004':	ocupacao_mae,
        'Q005':	pessoas_cohab,
        'Q006':	renda_familiar,
        'Q007':	empregada_dom,
        'Q008':	num_banheiros,
        'Q009':	num_quartos,
        'Q010':	num_carro,
        'Q011':	num_motocicleta,	
        'Q012':	num_geladeira,
        'Q013':	num_freezer,
        'Q014':	num_maqlavar,	
        'Q015':	num_maqsecar,
        'Q016':	num_microondas,
        'Q017':	num_lavalouca,
        'Q018':	aspirador,	
        'Q019':	num_televisao,	
        'Q020':	dvd,	
        'Q021':	tv_assinatura,	
        'Q022':	num_celular,
        'Q023':	tel_fixo,	
        'Q024':	num_computador,
        'Q025':	acesso_internet		
        }
    
fig, ax = plt.subplots(1,1, figsize=(10,20))
lgb.plot_importance(lgb_model, ax=ax)

st.pyplot(fig)