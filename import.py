import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

# テキストデータを読み込む
with open("data/input1.txt", "r") as f:
    docs = f.read()

# テキストを分割
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_text(docs)

# OpenAI埋め込みモデルを使用
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))

# ベクトルデータベースを作成
faiss_index = FAISS.from_texts(texts, embeddings)

# インデックスを保存
faiss_index.save_local("faiss_index")

