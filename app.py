from googlesearch import search
from newspaper import Article
import re
from transformers import pipeline, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm  # Progress bar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.llms import HuggingFacePipeline
from langchain_core.documents import Document
# This downloads and caches the model/tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def fetch_article(keyword, limit=5):
    """
    Fetches articles related to a keyword using Google search and extracts their content.
    """
    print(f"🔍 Searching for articles related to: {keyword}")
    query = (
        f"{keyword} site:ndtv.com OR site:indiatoday.in OR site:hindustantimes.com "
        f"OR site:thehindu.com OR site:timesofindia.indiatimes.com OR site:indianexpress.com "
        f"OR site:livemint.com OR site:business-standard.com OR site:moneycontrol.com OR site:republicworld.com"
    )

    articles = []
    urls = list(search(query, num_results=limit, lang='en'))

    print(f"📰 Fetching and parsing {len(urls)} articles...\n")
    for url in tqdm(urls, desc="Downloading articles"):
        try:
            article = Article(url)
            article.download()
            article.parse()
            articles.append({
                'title': article.title,
                'url': url,
                'content': article.text
            })
        except Exception as e:
            print(f"⚠️ Error processing {url}: {e}")

    print("✅ Finished fetching articles. Cleaning content...")

    content = ""
    for article in articles:
        content += article["content"] + "\n\n"

    # Clean text
    content = content.lower()
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'[^\w\s:\.\-]', '', content)
    content = content.strip()

    return content


def split(text):
    """
    Splits the text into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    print("\n✂️ Splitting text into chunks...")
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=lambda x: len(tokenizer.encode(x, truncation=False)),
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"📄 Generated {len(chunks)} chunks.")
    return chunks


# Example usage
chunks = split(fetch_article("Iran"))
for i, chunk in enumerate(chunks[:3]):
    print(f"\n🔹 Chunk {i+1}:\n{chunk[:300]}...")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
dim = len(embeddings.embed_query("example"))  # Get the dimension of the embeddings
print(f"🔍 Embedding dimension: {dim}")

index = faiss.IndexFlatL2(dim)  # Create a FAISS index
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={},
)
docs = [Document(page_content=chunk metadata={"url": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
vector_store.add_documents(docs)
print(f"📚 Added {len(docs)} chunks to the vector store.")

def search_similar(query, k=5):
    """
    Searches for similar chunks in the vector store based on the query.
    """
    print(f"🔎 Searching for similar chunks to: {query}")
    query_embedding = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=k)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n🔸 Result {i+1}:\n{result.page_content[:300]}... (URL: {result.metadata.get('url', 'N/A')})")
    
    return results

search_results = search_similar("Iran", k=3)
print("\n🔍 Search Results:" , search_results)

pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer=tokenizer,
    device=-1  # Use GPU if available, otherwise set to -1 for CPU
)
llm = HuggingFacePipeline(pipeline=pipe)

def summarize_text(chunks):
    """
    Performs map-reduce summarization:
    1. Summarizes each chunk individually (map step)
    2. Combines those summaries into a final summary (reduce step)
    """
    print("🧠 Step 1: Summarizing each chunk...")
    intermediate_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  - Chunk {i+1}/{len(chunks)}")
        
        # Truncate to safe length for BART
        input_ids = tokenizer.encode(chunk, truncation=True, max_length=1024)
        truncated_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)

        summary = llm.invoke(truncated_chunk)
        intermediate_summaries.append(summary)

    print("\n🧩 Step 2: Reducing intermediate summaries into a final summary...")
    combined_text = "\n\n".join(intermediate_summaries)

    # Final summary
    final_input_ids = tokenizer.encode(combined_text, truncation=True, max_length=1024)
    final_text = tokenizer.decode(final_input_ids, skip_special_tokens=True)
    final_summary = llm.invoke(final_text)

    return final_summary

print(summarize_text(chunks))