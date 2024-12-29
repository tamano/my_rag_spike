import os
from dotenv import load_dotenv

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

# OpenAI埋め込みモデルの設定
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# FAISSインデックスをロード
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# OpenAI Chat Model（LLM）を設定
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# 質問応答のチェーンをロード
qa_chain = load_qa_chain(llm, chain_type="stuff")

# RetrievalQA を設定
qa = RetrievalQA(
    combine_documents_chain=qa_chain,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# Streamlit UI
st.title("My ChatAI")
query = st.text_input("質問を入力してください")

if query:
    with st.spinner("回答を生成中..."):
        # qa.run() の代わりに qa() を使用
        result = qa({"query": query})

        # 回答の出力
        st.write("### 回答")
        st.write(result["result"])

        # 参照ドキュメントの出力
        st.write("### 参照ドキュメント")
        for doc in result["source_documents"]:
            st.write(doc.page_content)
