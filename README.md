# my_rag_spike
ChatGPTを用いたRAG開発の練習

## インストール

### チェックアウト
```
git clone git@github.com:tamano/my_rag_spike.git
```

### 必要モジュールの追加インストール
```
pip install streamlit
pip install python-dotenv
pip install langchain langchain-openai langchain-community faiss-cpu
```

### 準備
```
cp .env.sample .env

vim .env     # OpenAIのAPI Keyを記入する
```

## 使い方
1. `OPENAI_API_KEY` 環境変数に、OpenAIのAPIKEYを入れる。
1. `data/input1.txt` に学習させたい内容を入れる。
1. `python import.py` でベクターデータベースを作る。
1. `streamlit run app.py` で対話式UIを起動する。
