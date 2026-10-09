import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import platform

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO (STUDIO DARK MODE)
# ==========================================
st.set_page_config(
    page_title="Pre-Pro AI | Análisis de Guiones",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto Condensed', sans-serif;
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    
    .stTextInput input, .stTextArea textarea {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    
    h1, h2, h3 { color: #f0B326 !important; text-transform: uppercase;}
    
    .suggestion-box {
        background-color: #161b22;
        border-left: 4px solid #f0B326;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BARRA LATERAL (CONFIGURACIÓN)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3175/3175114.png", width=70)
    st.header("🎬 Set de Pre-Producción")
    st.write("Sube tu Guion Literario o Documento de Diseño (GDD).")
    
    # Manejo seguro de la API Key
    ke = st.text_input('🔑 Clave de OpenAI (API Key):', type="password")
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        
    st.divider()
    pdf = st.file_uploader("📄 Cargar Documento (PDF)", type="pdf")
    
    st.divider()
    st.caption(f"Entorno: Python {platform.python_version()}")

# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================
st.title('🎬 Pre-Pro AI: Desglose de Producción')
st.markdown("Asistente RAG para extraer requerimientos de modelado 3D, referencias de iluminación y listados de *props* directamente de tus guiones técnicos o conceptuales.")

if not ke:
    st.warning("⚠️ Ingresa tu API Key de OpenAI en el panel lateral para encender el motor neuronal.")
    st.stop()

if pdf is None:
    st.info("👈 Esperando el documento base (PDF). Por favor cárgalo en el panel lateral.")
    st.stop()

# ==========================================
# 4. PROCESAMIENTO RAG (VECTORIZACIÓN)
# ==========================================
try:
    with st.spinner("Procesando documento y generando embeddings espaciales..."):
        # 4.1. Extracción de texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # 4.2. Fragmentación (Chunking)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=700,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 4.3. Base de Datos Vectorial
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
    st.success(f"✅ Documento indexado correctamente ({len(chunks)} fragmentos). Listo para analizar.")

    # ==========================================
    # 5. MOTOR DE CONSULTAS Y PROMPTS SUGERIDOS
    # ==========================================
    col1, col2 = st.columns([2, 1])
    
    if 'pre_pro_query' not in st.session_state:
        st.session_state.pre_pro_query = ""

    with col2:
        st.write("### 💡 Consultas de Dirección de Arte")
        if st.button("📦 Lista de Props para Modelar (3D)"):
            st.session_state.pre_pro_query = "Extrae una lista de todos los objetos, muebles o elementos físicos mencionados que requieran ser modelados en 3D. Formatéalo como una lista de tareas."
            st.rerun()
        if st.button("💡 Desglose de Iluminación"):
            st.session_state.pre_pro_query = "Analiza el documento y describe qué tipo de iluminación (día, noche, oscura, neón) o atmósfera visual se menciona o sugiere en las escenas."
            st.rerun()
        if st.button("🕹️ Mecánicas / Interactividad"):
            st.session_state.pre_pro_query = "Si es un proyecto interactivo o VR, resume cuáles son las mecánicas principales o interacciones que el usuario/espectador puede realizar."
            st.rerun()

    with col1:
        user_question = st.text_area(
            "🔍 Pregunta libre al asistente de producción:", 
            value=st.session_state.pre_pro_query,
            height=120,
            placeholder="Ej: ¿Cuáles son las locaciones principales mencionadas en el guion?"
        )
        
        analizar = st.button("EJECUTAR ANÁLISIS 🚀", type="primary")

    if analizar and user_question:
        with st.spinner("Buscando referencias en el documento..."):
            # Búsqueda semántica
            docs = knowledge_base.similarity_search(user_question)
            
            # 5.1. Actualización al modelo de Chat moderno (gpt-4o-mini)
            llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
            
            # Cadena QA
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            st.markdown("---")
            st.markdown("### 📋 Reporte de Producción:")
            
            st.markdown(f"""
            <div class="suggestion-box">
                {response}
            </div>
            """, unsafe_allow_html=True)
            
except Exception as e:
    st.error(f"Error crítico en el análisis vectorial: {str(e)}")
