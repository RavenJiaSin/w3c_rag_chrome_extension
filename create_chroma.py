import sqlite3
from sentence_transformers import SentenceTransformer
import chromadb

# 初始化模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 初始化 ChromaDB 客戶端
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("w3c_standards")

# 連接到 SQLite 資料庫
conn = sqlite3.connect('w3c_data.db')
cursor = conn.cursor()

# 讀取資料
cursor.execute('SELECT title, content FROM w3c_standards')
rows = cursor.fetchall()

# 定義每個文本的最大長度（以字數為單位）
# 注意：SentenceTransformer 通常有 token 限制，而不是字數限制。
# MiniLM-L6-v2 的最大序列長度通常是 256 或 512 tokens。
# 以字數估算可能不精確，但作為粗略分割是可以的。
# 如果遇到模型截斷警告，可能需要更精確的 tokenizer 來分割。
MAX_TEXT_LENGTH = 1500 # 稍微保守一點，避免超過 token 限制太多

# 分割長文本 (基於字數)
def split_text_by_words(text, max_words=MAX_TEXT_LENGTH):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        # 檢查加入這個詞是否會超過長度
        # 加 1 是為了空格
        if current_length + len(word) + (1 if current_chunk else 0) > max_words:
            if current_chunk: # 如果當前 chunk 不為空，先儲存
                chunks.append(" ".join(current_chunk))
            current_chunk = [word] # 新的 chunk 從這個詞開始
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + (1 if len(current_chunk) > 1 else 0) # 更新長度
    if current_chunk: # 加入最後一個 chunk
        chunks.append(" ".join(current_chunk))
    return chunks


# 批次處理以提高效率
batch_size = 100 # 可以根據你的記憶體調整
all_documents = []
all_metadatas = []
all_embeddings = []
all_ids = []
doc_id_counter = 0

print("Starting data processing and embedding...")

for i, row in enumerate(rows):
    title = row[0]
    content = row[1] if row[1] else "" # 確保 content 不是 None

    if not content.strip(): # 跳過空的 content
        print(f"Skipping row {i} with empty content for title: {title}")
        continue

    # 先將內容分割成多個部分
    text_chunks = split_text_by_words(content)

    if not text_chunks: # 如果分割後是空的 (例如 content 只有空格)
        print(f"Skipping row {i} after splitting resulted in empty chunks for title: {title}")
        continue

    # 處理每個分割的片段
    chunk_embeddings = model.encode(text_chunks).tolist() # 一次性編碼所有 chunks 並轉為 list

    for j, chunk in enumerate(text_chunks):
        doc_id = f"doc_{i}_chunk_{j}"
        all_ids.append(doc_id)
        # 儲存文件和元數據
        # 注意：如果原始 content 很長，metadata 也會很大，可能影響查詢性能或儲存
        # 可以考慮只儲存 title 或 chunk 本身作為 metadata
        all_documents.append(f"Title: {title} Content: {chunk}")
        all_metadatas.append({"title": title, "original_content_preview": content[:200] + "..."}) # 儲存部分原文預覽
        all_embeddings.append(chunk_embeddings[j]) # 加入對應的 embedding

        doc_id_counter += 1

        # 達到批次大小或處理完所有 row 時，添加到 ChromaDB
        if len(all_ids) >= batch_size or (i == len(rows) - 1 and j == len(text_chunks) - 1):
            if all_ids: # 確保列表不為空
                print(f"Adding batch of {len(all_ids)} items to ChromaDB...")
                try:
                     collection.add(
                         ids=all_ids,
                         documents=all_documents,
                         metadatas=all_metadatas,
                         embeddings=all_embeddings
                     )
                     print(f"Successfully added batch. Total items processed: {doc_id_counter}")
                     # 清空列表以準備下一批次
                     all_documents = []
                     all_metadatas = []
                     all_embeddings = []
                     all_ids = []
                except Exception as e:
                    print(f"Error adding batch to ChromaDB: {e}")
                    # 可以選擇在這裡停止、記錄錯誤或跳過批次
                    # 清空列表以避免重複嘗試錯誤的批次
                    all_documents = []
                    all_metadatas = []
                    all_embeddings = []
                    all_ids = []


print("Finished processing and adding data to ChromaDB.")

# 關閉資料庫連接
conn.close()

print("Database connection closed.")
# 確保 ChromaDB 持久化
# client.persist() # 在較新版本中，如果設定了 persist_directory，通常會自動處理，但顯式調用也可以
print("ChromaDB changes should be persisted.")