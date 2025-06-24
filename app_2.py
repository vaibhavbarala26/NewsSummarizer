import streamlit as st
from googlesearch import search
from newspaper import Article
import re
from transformers import pipeline, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

# ---------- PAGE CONFIG ---------- #
st.set_page_config(
    page_title="NewsAI",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS ---------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    .main, .block-container {
        background: transparent !important;
        padding: 2rem 3rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem !important;
        color: #64748b !important;
        font-weight: 400 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #e2e8f0 !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(10px) !important;
}

/* Set placeholder color to black */
input::placeholder {
     color: #64748b !important;
    opacity: 1 !important; /* Required for Firefox */
}

    
    .stTextInput > div > div > input:focus {
        border-color: #8E2DE2 !important;
        box-shadow: 0 0 0 3px rgba(142, 45, 226, 0.1) !important;
        color: black !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(74, 0, 224, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(74, 0, 224, 0.4) !important;
    }
    
    /* Card Styling */
    .card {
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.8) !important;
        padding: 1.5rem !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08) !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(226, 232, 240, 0.6) !important;
        transition: all 0.2s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.12) !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    .card h4 {
        color: #1e293b !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.4 !important;
    }
    
    .card a {
        color: #8E2DE2 !important;
        text-decoration: none !important;
        font-size: 0.9rem !important;
    }
    
    .card a:hover {
        color: #4A00E0 !important;
        text-decoration: underline !important;
    }
    
    .card p {
        color: #374151 !important;
        line-height: 1.7 !important;
        font-size: 1rem !important;
        margin: 0 !important;
    }
    
    /* Section Headers */
    h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        font-size: 1.4rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center !important;
        color: #64748b !important;
        font-size: 0.9rem !important;
        margin-top: 3rem !important;
        padding: 2rem 0 !important;
        border-top: 1px solid rgba(226, 232, 240, 0.6) !important;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #8E2DE2 !important;
        color:black !important;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main, .block-container {
            padding: 1rem 1.5rem !important;
        }
        
        .main-title {
            font-size: 2rem !important;
        }
        
        .subtitle {
            font-size: 1rem !important;
        }
        
        .card {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------- MODEL SETUP ---------- #
@st.cache_resource
def load_model():
    MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pipe = pipeline("summarization", model=MODEL_NAME, tokenizer=tokenizer, device=-1)
    llm = HuggingFacePipeline(pipeline=pipe)
    return tokenizer, llm

tokenizer, llm = load_model()

# ---------- FUNCTIONS ---------- #
def fetch_articles(keyword, limit=5):
    """Fetch articles from Indian news sources"""
    query = (
        f"{keyword} site:ndtv.com OR site:indiatoday.in OR site:hindustantimes.com "
        f"OR site:thehindu.com OR site:timesofindia.indiatimes.com OR site:indianexpress.com "
        f"OR site:livemint.com OR site:business-standard.com OR site:moneycontrol.com OR site:republicworld.com"
    )
    urls = list(search(query, num_results=limit, lang='en'))

    articles = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if len(article.text) > 100:  # Only include articles with substantial content
                articles.append({
                    "title": article.title or "Untitled Article",
                    "url": url,
                    "content": article.text
                })
        except Exception:
            continue

    if not articles:
        return "", []

    full_text = "\n\n".join([a["content"] for a in articles]).lower()
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = re.sub(r'[^\w\s:\.\-]', '', full_text)
    return full_text.strip(), articles

def split_text(text):
    """Split text into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=lambda x: len(tokenizer.encode(x, truncation=False)),
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def create_documents(chunks, metadata):
    """Create document objects from text chunks"""
    return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

def build_vector_store(documents):
    """Build FAISS vector store for similarity search"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)
    store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})
    store.add_documents(documents)
    return store, embeddings

def search_similar_chunks(query, vector_store, embeddings, k=3):
    """Search for similar chunks using vector similarity"""
    query_vector = embeddings.embed_query(query)
    return vector_store.similarity_search_by_vector(query_vector, k=k)

def summarize_chunks(chunks):
    """Generate summary from text chunks"""
    summaries = []
    for chunk in chunks:
        input_ids = tokenizer.encode(chunk, truncation=True, max_length=1024)
        truncated = tokenizer.decode(input_ids, skip_special_tokens=True)
        try:
            summary = llm.invoke(truncated)
            summaries.append(summary)
        except Exception:
            continue
    
    if not summaries:
        return "Unable to generate summary from the retrieved content."
    
    combined = "\n".join(summaries)
    input_ids = tokenizer.encode(combined, truncation=True, max_length=1024)
    final_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    try:
        return llm.invoke(final_text)
    except Exception:
        return summaries[0] if summaries else "Unable to generate summary."

# ---------- UI LAYOUT ---------- #

# Header Section
st.markdown('<h1 class="main-title">🧠 NewsAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get concise AI-generated summaries from India\'s top news outlets</p>', unsafe_allow_html=True)

# Search Section
keyword = st.text_input("", placeholder="Try 'Lok Sabha 2024', 'Cricket World Cup', or 'Stock Market'", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    search_button = st.button("🚀 Search & Summarize")

# Main Content
if search_button and keyword:
    with st.spinner("🔎 Collecting and summarizing..."):
        try:
            content, articles = fetch_articles(keyword)
            
            if not articles:
                st.error("❌ No valid articles found for your search query. Please try different keywords.")
            else:
                chunks = split_text(content)
                if chunks:
                    docs = create_documents(chunks, metadata={"title": articles[0]['title'], "url": articles[0]['url']})
                    vector_store, embeddings = build_vector_store(docs)
                    top_chunks = search_similar_chunks(keyword, vector_store, embeddings)
                    final_summary = summarize_chunks([doc.page_content for doc in top_chunks])

                    # Create two columns for layout
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("### 📰 Source Articles")
                        for article in articles:
                            st.markdown(f"""
                                <div class='card'>
                                    <h4>{article['title']}</h4>
                                    <a href="{article['url']}" target="_blank">🔗 Read full article</a>
                                </div>
                            """, unsafe_allow_html=True)

                    with col_b:
                        st.markdown("### 📌 AI Summary")
                        st.markdown(f"""
                            <div class='card'>
                                <p>{final_summary.strip()}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Unable to process the retrieved content for summarization.")
                    
        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")

elif search_button and not keyword:
    st.warning("⚠️ Please enter a keyword to search for news articles.")

# Footer
st.markdown('<div class="footer">🔎 Built with ❤️ using Streamlit • Summarizing Indian news from NDTV, IE, HT, and more</div>', unsafe_allow_html=True)