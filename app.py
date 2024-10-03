import streamlit as st
import whisper
import librosa
import numpy as np
import os
import io
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings, Document
from llama_index.llms.groq import Groq

# Carregar modelo Whisper
model = whisper.load_model("base")

# Configurar LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
key=os.getenv('GROQ_API_KEY')
# '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'
Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=key) #llama3-70b-8192

# Templates para LlamaIndex
text_qa_template_str = (
    "Você é um assistente que está analisando a transcrição de uma sessão de RPG. O mestre não conseguiu anotar tudo, "
    "então sua tarefa é compilar as ações mais importantes realizadas pelos personagens principais, cujos nomes são {character_names}. "
    "Os outros nomes que aparecem são NPCs controlados pelo mestre.\n"
    "Abaixo está a transcrição da sessão de RPG:\n---------------------\n{context_str}\n---------------------\n"
    "Com base nessa transcrição, forneça um resumo detalhado das seguintes ações dos personagens principais:\n"
    "- Onde eles estão\n"
    "- Quem eles atacaram ou com quem lutaram\n"
    "- Quais combates ocorreram e os resultados\n"
    "- O que compraram ou obtiveram\n"
    "- NPCs com quem interagiram e o que discutiram\n"
    "- Qualquer outro evento importante que tenha ocorrido e envolva os personagens principais\n"
    "Se o contexto não fornecer informações suficientes, faça o seu melhor com base no que está disponível.\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "A pergunta original é a seguinte: {query_str}\n"
    "Aqui está a resposta atual que foi gerada: {existing_answer}\n"
    "Temos a oportunidade de refinar essa resposta com mais contexto abaixo:\n"
    "------------\n{context_msg}\n------------\n"
    "Utilizando o novo contexto e as suas informações já geradas, por favor, refine ou repita a resposta existente. "
    "Certifique-se de incluir mais detalhes sobre as ações dos personagens principais, como suas localizações, interações com NPCs, combates, compras ou outros eventos importantes.\n"
    "Se o contexto não fornecer informações suficientes, faça o seu melhor com base no que está disponível.\n"
)
refine_template = PromptTemplate(refine_template_str)

# Função para carregar o áudio e adaptá-lo ao Whisper
def prepare_audio_for_whisper(audio_file):
    # Carregar o arquivo MP3 como um array usando librosa
    y, sr = librosa.load(io.BytesIO(audio_file.read()), sr=None)
    
    # Resample para 16kHz, se necessário
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    # Normalizar o áudio entre -1.0 e 1.0
    y = np.clip(y / np.max(np.abs(y)), -1.0, 1.0)
    
    return y

# Função para processar áudio MP3 e gerar transcrição e resumo
def process_audio(audio_file, character_names):
    # Preparar o áudio para Whisper (converter em numpy array 16kHz)
    audio_numpy = prepare_audio_for_whisper(audio_file)

    # Transcrever o áudio usando Whisper
    result = model.transcribe(audio_numpy, language="pt")
    transcription = result["text"]
    
    # Processar a transcrição no LlamaIndex
    # documents = [transcription]  # Inserir diretamente o texto
    document = Document(text = transcription)
    index = VectorStoreIndex.from_documents([document])
    
    # Criar o motor de consulta
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
    )
    
    # Formatar a consulta com os nomes dos personagens fornecidos
    query = (
        f"Quais foram as ações mais importantes realizadas por {character_names} durante a sessão, "
        "incluindo onde estão, quem atacaram, combates feitos, o que compraram, e NPCs com quem interagiram?"
    )
    
    # Fazer a consulta sobre as ações dos personagens
    response = query_engine.query(query)
    
    # Retornar a transcrição e o resumo gerado
    return transcription, response

# Interface do Streamlit
st.title("Assistente de GM")

# Upload de arquivo de áudio MP3
uploaded_audio = st.file_uploader("Carregue o áudio MP3", type=["mp3"])

# Campo para inserir os nomes dos personagens
character_names = st.text_input("Insira os nomes dos personagens principais (separados por vírgula)")

if uploaded_audio is not None and character_names:
    # Processar o áudio e gerar transcrição e resumo
    transcription, summary = process_audio(uploaded_audio, character_names)
    
    # Exibir a transcrição e o resumo no Streamlit
    st.subheader("Transcrição")
    st.text_area("Transcrição", transcription, height=300)
    
    st.subheader("Resumo das Ações dos Personagens")
    st.text_area("Resumo", summary, height=30000)