from flask import Flask, request, jsonify
from flask_cors import CORS # 用於處理來自 Chrome 擴充套件的請求
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import os # 檢查資料庫路徑

# --- 設定 ---
DB_PATH = "db"
COLLECTION_NAME = "w3c_standards"
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
N_RESULTS = 3 # 指定要檢索多少個相關的文本區塊

# --- 初始化 ---
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# 取得你的 Chrome 擴充套件 ID (可以在 chrome://extensions/ 找到)
extension_id = "odmikollhnahmfohmdafkpnlfpopicmh"
cors_origin = f"chrome-extension://{extension_id}"

# 允許來自指定 Chrome 擴充套件的請求
CORS(app, resources={r"/search": {"origins": cors_origin}})
# 如果在本地開發時需要更寬鬆的測試（安全性較低）：
# CORS(app) # 允許所有來源，僅限測試

# --- 載入模型和資料庫 ---
try:
    print("正在載入 Sentence Transformer 模型...")
    if not os.path.exists(DB_PATH):
         print(f"錯誤：資料庫路徑 '{DB_PATH}' 不存在。請先運行 create_chroma.py。")
         exit()
    model = SentenceTransformer(MODEL_NAME)
    print("模型載入完成。")

    print("正在連接到 ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    # 檢查集合是否存在
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"已連接到 ChromaDB 集合 '{COLLECTION_NAME}'。項目數量: {collection.count()}")
    except Exception as e:
        print(f"錯誤：無法獲取集合 '{COLLECTION_NAME}'。請確認它已透過 create_chroma.py 創建。錯誤訊息: {e}")
        exit()

except Exception as e:
    logging.error(f"初始化過程中發生錯誤: {e}", exc_info=True)
    # 如果核心元件載入失敗則退出
    exit("模型或資料庫初始化失敗，程式結束。")


# --- API 端點 ---
@app.route('/search', methods=['POST'])
def search():
    # 檢查請求是否為 JSON 格式
    if not request.is_json:
        return jsonify({"error": "請求必須是 JSON 格式"}), 400

    data = request.get_json()
    question = data.get('question') # 從請求的 JSON 中獲取 'question'

    # 檢查是否有提供問題
    if not question:
        return jsonify({"error": "請求內容中缺少 'question'"}), 400

    logging.info(f"收到問題: {question}")

    try:
        # 1. 將問題編碼成向量
        question_embedding = model.encode(question).tolist()

        # 2. 查詢 ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding], # 查詢向量列表
            n_results=N_RESULTS,                  # 返回結果數量
            include=['documents']                 # 我們只需要文件內容作為上下文
        )

        # 3. 提取並組合上下文
        # results['documents'] 是一個列表，其中包含一個列表（因為我們只查詢了一個向量）
        retrieved_docs = results.get('documents', [[]])[0] # 安全地獲取文件列表

        if not retrieved_docs:
             logging.warning("在 ChromaDB 中未找到相關文件。")
             # 如果找不到，可以返回一個提示訊息，或讓 LLM 自行處理
             context = "在提供的 W3C 標準資料庫中找不到相關內容。"
        else:
            # 將檢索到的文本區塊組合成單一的上下文文字串
            # 使用分隔符讓 LLM 容易區分不同的來源區塊
            context = "\n\n---\n\n".join(retrieved_docs)
            logging.info(f"已檢索 {len(retrieved_docs)} 個文件作為上下文。")
            # 打印檢索到的文件片段（用於除錯，可選）
            # for i, doc in enumerate(retrieved_docs):
            #    logging.debug(f"上下文文件 {i+1}: {doc[:100]}...")

        # 4. 返回包含上下文的 JSON 回應
        return jsonify({"context": context})

    except Exception as e:
        # 記錄處理請求時發生的任何錯誤
        logging.error(f"處理搜尋請求時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "搜尋過程中發生內部伺服器錯誤"}), 500

# --- 運行伺服器 ---
if __name__ == '__main__':
    print("正在啟動 Flask 伺服器於 http://127.0.0.1:5000")
    print(f"將允許來自 Chrome 擴充套件 ({cors_origin}) 的請求")
    # 使用 host='0.0.0.0' 如果你需要從區域網路內的其他設備訪問
    # debug=True 僅用於開發，正式部署時應設為 False
    app.run(host='127.0.0.1', port=5000, debug=False)