import os
from dotenv import load_dotenv

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# OpenAI埋め込みモデルの設定
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# FAISSインデックスをロード
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# OpenAI Chat Model（LLM）を設定
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Retriever を設定
retriever = db.as_retriever()

# プロンプトテンプレート
template = """以下のコンテキストを使用して質問に答えてください。

コンテキスト: {context}

質問: {question}

回答:"""

prompt = PromptTemplate.from_template(template)

# LCELチェーンを作成
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("My ChatAI")
query = st.text_input("質問を入力してください")

if query:
    with st.spinner("回答を生成中..."):
        # 回答を生成
        result = rag_chain.invoke(query)
        
        # 参照ドキュメントを取得
        docs = retriever.invoke(query)

        # 回答の出力
        st.write("### 回答")
        st.write(result)

        # 参照ドキュメントの出力
        st.write("### 参照ドキュメント")
        for doc in docs:
            st.write(doc.page_content)
