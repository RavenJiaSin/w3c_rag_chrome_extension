# -*- coding: utf-8 -*-
# ==============================================================================
# RAG Server using Flask (Synchronous with Background History Update)
# ==============================================================================
import logging
import os
import re
import time
from typing import Optional, List, Dict
import json
import chromadb
import faiss
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from groq import Groq, RateLimitError, APIError
from langdetect import detect, LangDetectException
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
import textwrap
import threading

# --- 1. Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
logging.info("環境變數已載入 (Environment variables loaded).")

# --- Paths and Model Names ---
EN_EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
ZH_EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-chinese'
MODEL_FALLBACK_LIST = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
]
FAISS_BASE_DIR = "faiss_w3c_stores"
CHROMA_BASE_DIR = "chroma_w3c_stores"
MESSAGE_HISTORY_COLLECTION_NAME = "conversation_history"
FIELDS_TO_VECTORIZE = ['abstract', 'status_of_document', 'content_summary', 'original_content_snippet']
LANGUAGES = ['en', 'zh']
MAX_CONTEXT_TOKENS = 4000
TOP_K_CONTENT = 2
TOP_K_HISTORY = 3
SERVER_PORT = 5050 # Use the working port

# --- Global Application State ---
app_state = {} # Dictionary to hold loaded resources

# --- 2. Pydantic Models for Request Validation ---
class RAGRequest(BaseModel):
    """Validates the incoming request body."""
    query_message: str
    model_name: str

# --- 3. Helper Functions (Synchronous) ---
def get_faiss_paths(base_dir: str, field_name: str, lang: str) -> tuple[str, str]:
    """Generates standardized file paths for FAISS index and metadata."""
    filename_base = f"{field_name}_{lang}"
    index_path = os.path.join(base_dir, f"{filename_base}.index")
    metadata_path = os.path.join(base_dir, f"{filename_base}_meta.json")
    logging.debug(f"Generated FAISS paths: Index='{index_path}', Meta='{metadata_path}'")
    return index_path, metadata_path

def get_chroma_collection_name(field_name: str, lang: str) -> str:
    """Generates a standardized collection name for ChromaDB."""
    collection_name = f"w3c_{field_name}_{lang}"
    logging.debug(f"Generated Chroma collection name: {collection_name}")
    return collection_name

def count_chars(text: str) -> tuple[int, int]:
    """Counts English letters and Chinese characters in a string."""
    # Count English letters (case-insensitive)
    en_count = len(re.findall(r'[a-zA-Z]', text))
    # Count Chinese characters (basic CJK Unified Ideographs range)
    zh_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    return en_count, zh_count

def detect_language(text: str) -> str:
    """
    Detects language ('en' or 'zh') using langdetect first, then falls back
    to character counting for ambiguous cases. Defaults to 'en'.
    """
    # --- Step 1: Try langdetect first ---
    detected_lang = None
    try:
        lang = detect(text)
        if lang.startswith('zh'):
            detected_lang = 'zh'
        elif lang == 'en':
            detected_lang = 'en'
        # else: lang is something else, proceed to manual check
        logging.debug(f"langdetect result: {lang}") # Log initial detection
    except LangDetectException:
        logging.warning("Language detection by langdetect failed, falling back to char count.")
        # Proceed to manual check

    # If langdetect gave a clear en/zh result, trust it (usually reliable for longer, clear texts)
    if detected_lang:
        logging.info(f"Language detected by langdetect: {detected_lang}")
        return detected_lang

    # --- Step 2: Fallback to character counting for ambiguous/failed cases ---
    logging.info("langdetect inconclusive, performing character count analysis...")
    en_count, zh_count = count_chars(text)
    logging.debug(f"Character counts: EN={en_count}, ZH={zh_count}")

    # --- Step 3: Decision Logic based on counts ---
    # If Chinese characters are present and significantly outnumber English letters
    # Adjust the threshold (e.g., zh_count > 1, zh_count > en_count * 0.5) as needed
    if zh_count > 0 and zh_count >= en_count: # Simple heuristic: more or equal Chinese chars -> likely Chinese
        final_lang = 'zh'
    # If English letters significantly outnumber Chinese characters, or only English exists
    elif en_count > zh_count:
         final_lang = 'en'
    # If counts are equal (and non-zero), or both zero, default might be needed, or use langdetect's initial guess if available
    # For simplicity, default to English if unclear after counting
    else:
        logging.warning("Character count analysis inconclusive (counts equal or both zero), defaulting to 'en'.")
        final_lang = 'en'

    logging.info(f"Language determined by character count: {final_lang}")
    return final_lang

def format_context(results: List[Dict]) -> str:
    """Formats retrieved search results into a single context string for LLM."""
    context_parts = []
    for res in results:
        metadata=res.get('metadata',{}); doc_text=res.get('document',''); title=metadata.get('title','N/A'); source_field=metadata.get('source_field','N/A'); role=metadata.get('role'); source_db=res.get('source_db','Unknown')
        if role: prefix = f"--- History ({role}) [{source_db}] ---"
        else: prefix = f"--- Context from {source_field} (Title: {title}) [{source_db}] ---"
        content_to_add = doc_text if doc_text else f"Retrieved item with Title: {title}"
        context_parts.append(f"{prefix}\n{content_to_add}")
    return "\n\n".join(context_parts).strip()

# --- Synchronous LLM Call Function ---
def call_groq_llm_sync(client: Groq, prompt: str, target_model: str) -> Optional[str]:
    """Calls Groq LLM API synchronously with model fallback."""
    models_to_try = [target_model] + [m for m in MODEL_FALLBACK_LIST if m != target_model]
    last_error = None
    for model_id in models_to_try:
        logging.debug(f"Attempting LLM generation with model: {model_id}")
        try:
            # Direct synchronous call
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                model=model_id,
                temperature=0.5,
                max_tokens=1024
            )
            response = chat_completion.choices[0].message.content.strip()
            if response:
                logging.info(f"LLM success: {model_id}")
                return response
            else:
                logging.warning(f"{model_id} empty response.")
                last_error="Empty response"; continue
        except RateLimitError as e:
            logging.warning(f"Rate limit {model_id}. Trying next...")
            last_error=e; time.sleep(1); continue # Use time.sleep
        except APIError as e_api:
            logging.error(f"API error {model_id}: {e_api}")
            last_error=e_api; break # Stop on non-5xx errors
        except Exception as e:
            logging.error(f"Unexpected LLM error {model_id}: {e}")
            last_error=e; break
    logging.error(f"LLM failed. Last error: {last_error}")
    return None

# --- Synchronous Query Functions ---
def query_chroma_sync(collection, query_vector, k):
    """Synchronously queries a ChromaDB collection."""
    try:
        # Direct synchronous call
        return collection.query(
            query_embeddings=[query_vector.tolist()], # query expects list of embeddings
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
    except Exception as e_query:
        logging.error(f"Sync ChromaDB query failed for '{collection.name}': {e_query}")
        return None # Return None on error

def query_faiss_sync(index, metadata_list, query_vector, k):
    """Synchronously searches a FAISS index."""
    try:
        # FAISS search expects a 2D numpy array (n_queries, dimension)
        query_vec_np = query_vector.reshape(1, -1).astype('float32')
        # Direct synchronous call
        distances, indices = index.search(query_vec_np, k)
        results = []
        if indices.size > 0:
            for i in range(min(k, len(indices[0]))):
                idx = indices[0][i]
                dist = distances[0][i]
                if 0 <= idx < len(metadata_list):
                    meta = metadata_list[idx]
                    results.append({
                        "id": f"faiss_{idx}",
                        "distance": float(dist),
                        "metadata": meta,
                        "document": meta.get("title", "N/A") + " (FAISS)" # Placeholder doc
                    })
                else:
                    logging.warning(f"FAISS invalid idx:{idx}")
        return results
    except Exception as e_query:
        logging.error(f"Sync FAISS search failed: {e_query}")
        return [] # Return empty list on error

# --- Function to update history in a background thread ---
def update_history_background(history_collection, query_message, llm_response):
    """Embeds and upserts query/response to ChromaDB history in a separate thread."""
    try:
        # Access global state safely (assuming loaded)
        if "en_embedding_model" not in app_state or history_collection is None:
             logging.error("History Update: Required resources (embedding model or collection) not found.")
             return
        history_embedding_model = app_state["en_embedding_model"]

        logging.info("Background History Update: Embedding query and response...")
        # Perform blocking embedding in this thread
        query_embedding_hist = history_embedding_model.encode([query_message])
        response_embedding_hist = history_embedding_model.encode([llm_response])
        query_embed_list = query_embedding_hist[0].tolist() if query_embedding_hist is not None and len(query_embedding_hist)>0 else None
        response_embed_list = response_embedding_hist[0].tolist() if response_embedding_hist is not None and len(response_embedding_hist)>0 else None

        if query_embed_list and response_embed_list:
            timestamp=time.time(); user_query_id=f"hist_user_{timestamp}"; assistant_response_id=f"hist_asst_{timestamp}"
            # Perform blocking upsert
            history_collection.upsert(
                ids=[user_query_id, assistant_response_id],
                embeddings=[query_embed_list, response_embed_list],
                documents=[query_message, llm_response],
                metadatas=[{"role": "user", "timestamp": timestamp},{"role": "assistant", "timestamp": timestamp}]
            )
            logging.info("Background History Update: Update successful.")
        else:
            logging.error("Background History Update: Failed to generate embeddings.")
    except Exception as e:
        logging.error(f"Background History Update Error: {e}")

# --- 4. Global Resource Loading (Synchronous) ---
logging.info("腳本啟動：正在全局同步載入資源...")
resources_loaded = True
start_load_time = time.time()

# 4.1 Groq Client
logging.info("  (1/4) 正在初始化 Groq Client...")
try:
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key: raise ValueError("未在環境變數中找到 GROQ_API_KEY")
    app_state["groq_client"] = Groq(api_key=groq_api_key)
    logging.info("  (1/4) Groq Client 初始化成功。")
except Exception as e: logging.critical(f"  (1/4) 初始化 Groq Client 失敗: {e}"); resources_loaded = False

# 4.2 Embedding Models
if resources_loaded:
    logging.info("  (2/4) 正在載入 Embedding Models...")
    try:
        logging.info(f"    正在載入 EN Model: {EN_EMBEDDING_MODEL_NAME}")
        app_state["en_embedding_model"] = SentenceTransformer(EN_EMBEDDING_MODEL_NAME)
        logging.info(f"    EN Model 載入完成。")
        logging.info(f"    正在載入 ZH Model: {ZH_EMBEDDING_MODEL_NAME}")
        app_state["zh_embedding_model"] = SentenceTransformer(ZH_EMBEDDING_MODEL_NAME)
        logging.info("  (2/4) Embedding Models 載入成功。")
    except Exception as e: logging.critical(f"  (2/4) 載入 Embedding Models 失敗: {e}"); resources_loaded = False

# 4.3 ChromaDB Client and Collections
if resources_loaded:
    logging.info(f"  (3/4) 正在初始化 ChromaDB Client (路徑: {CHROMA_BASE_DIR})...")
    app_state["chroma_collections"] = {}
    app_state["chroma_ef_map"] = {
        'en': chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EN_EMBEDDING_MODEL_NAME),
        'zh': chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=ZH_EMBEDDING_MODEL_NAME)
    }
    try:
        os.makedirs(CHROMA_BASE_DIR, exist_ok=True)
        app_state["chroma_client"] = chromadb.PersistentClient(path=CHROMA_BASE_DIR)
        logging.info("    ChromaDB Client 初始化成功。正在載入 Collections...")
        available_collection_names = set()
        try:
            collection_objects = app_state["chroma_client"].list_collections()
            available_collection_names = {c.name for c in collection_objects}
        except Exception as e_list: logging.error(f"    列出 Chroma Collections 時出錯: {e_list}")

        loaded_chroma_count = 0
        for field in FIELDS_TO_VECTORIZE:
            for lang in LANGUAGES:
                collection_name = get_chroma_collection_name(field, lang); key = f"{field}_{lang}"
                if collection_name in available_collection_names:
                    try:
                        collection = app_state["chroma_client"].get_collection(name=collection_name, embedding_function=app_state["chroma_ef_map"][lang])
                        app_state["chroma_collections"][key] = collection; loaded_chroma_count += 1
                        logging.debug(f"      成功載入 Chroma Collection: {collection_name}")
                    except Exception as e_coll: logging.error(f"      無法載入 Chroma Collection '{collection_name}': {e_coll}"); app_state["chroma_collections"][key] = None
                else: logging.warning(f"      Chroma Collection '{collection_name}' 不存在。"); app_state["chroma_collections"][key] = None

        history_ef = app_state["chroma_ef_map"]['en']
        try:
            app_state["message_history_collection"] = app_state["chroma_client"].get_or_create_collection(name=MESSAGE_HISTORY_COLLECTION_NAME, embedding_function=history_ef, metadata={"hnsw:space": "l2"})
            logging.info("    Chroma Message History Collection 準備就緒。"); loaded_chroma_count += 1
        except Exception as e_hist: logging.error(f"    創建/載入 Message History Collection 失敗: {e_hist}"); app_state["message_history_collection"] = None
        logging.info(f"  (3/4) ChromaDB Collections 處理完成 ({loaded_chroma_count} 個成功)。")
    except Exception as e: logging.critical(f"  (3/4) 初始化 ChromaDB 失敗: {e}"); resources_loaded = False

# 4.4 Load FAISS Indices and Metadata
if resources_loaded:
    logging.info(f"  (4/4) 正在載入 FAISS 索引和元資料 (來自: {FAISS_BASE_DIR})...")
    app_state["faiss_indices"] = {}; app_state["faiss_metadata"] = {}
    faiss_loaded_count = 0
    try:
        os.makedirs(FAISS_BASE_DIR, exist_ok=True)
        for field in FIELDS_TO_VECTORIZE:
            for lang in LANGUAGES:
                key = f"{field}_{lang}"; idx_path, meta_path = get_faiss_paths(FAISS_BASE_DIR, field, lang)
                if os.path.exists(idx_path) and os.path.exists(meta_path):
                    try:
                        logging.debug(f"    正在載入 FAISS Index: {idx_path}")
                        app_state["faiss_indices"][key] = faiss.read_index(idx_path)
                        logging.debug(f"    正在載入 FAISS Metadata: {meta_path}")
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            app_state["faiss_metadata"][key] = json.load(f)
                        index_obj = app_state["faiss_indices"][key]; meta_list = app_state["faiss_metadata"][key]
                        logging.info(f"    成功載入 FAISS: {key} (Index: {index_obj.ntotal}, Meta: {len(meta_list)})")
                        if index_obj.ntotal != len(meta_list): logging.warning(f"      FAISS size mismatch for {key}!")
                        faiss_loaded_count += 1
                    except Exception as e_faiss_load: logging.error(f"    無法載入 FAISS Index/Meta for '{key}': {e_faiss_load}"); app_state["faiss_indices"][key] = None; app_state["faiss_metadata"][key] = None
                else: logging.warning(f"    FAISS Index/Meta 文件未找到 for '{key}'."); app_state["faiss_indices"][key] = None; app_state["faiss_metadata"][key] = None
    except Exception as e: logging.critical(f"  (4/4) 載入 FAISS 資料時發生嚴重錯誤: {e}"); resources_loaded = False
    logging.info(f"  (4/4) FAISS 索引和元資料載入完成 ({faiss_loaded_count} 個成功)。")

# 4.5 Final Check and Log
app_state["resources_ready"] = resources_loaded # Set readiness flag
if resources_loaded:
     end_load_time = time.time()
     logging.info(f"--- 所有資源全局載入完成，總耗時: {(end_load_time - start_load_time):.2f} 秒。---")
else:
     logging.critical("--- 資源載入失敗，伺服器可能無法正常工作。 ---")

# --- 5. Flask App Initialization ---
app = Flask(__name__) # Create Flask app instance
logging.info("Flask 應用實例已創建。")

# --- 6. API Endpoint Definition ---
@app.route("/rag_query", methods=["POST"])
def perform_rag_flask():
    """API endpoint using Flask to handle RAG requests."""
    request_start_time = time.time()
    try:
        # --- Log request entry and resource status ---
        logging.info(f"--- Endpoint Start (/rag_query) ---")
        resources_ready_flag = app_state.get("resources_ready", False) # Check flag safely
        logging.info(f"Endpoint check: resources_ready = {resources_ready_flag}")
        logging.debug(f"Current app_state keys: {list(app_state.keys())}") # Log keys for debug

        # --- Dependency Check ---
        if not resources_ready_flag: # Check the flag set during loading
            logging.error("伺服器資源未就緒。拒絕請求。")
            return jsonify({"detail": "Server resources not ready"}), 503 # Service Unavailable

        # Validate request data using Pydantic model
        try:
            # Get JSON data from Flask request object
            request_json = request.get_json()
            if not request_json:
                 return jsonify({"detail": "Request body must be JSON"}), 400
            # Validate data against the Pydantic model
            request_data = RAGRequest.model_validate(request_json)
        except ValidationError as e:
             logging.error(f"請求體驗證失敗: {e}")
             return jsonify({"detail": e.errors()}), 400 # Bad Request
        except Exception as e:
             logging.error(f"解析請求 JSON 或驗證時出錯: {e}")
             return jsonify({"detail": "Invalid JSON request body or validation error"}), 400

        query_message = request_data.query_message
        target_model = request_data.model_name
        logging.info(f"收到請求: query='{query_message[:50]}...', model='{target_model}'")

        # --- 1. Detect Language ---
        lang = detect_language(query_message)
        logging.info(f"Detected language: {lang}")

        # --- 2. Embed Query ---
        query_embedding_start_time = time.time()
        try:
            embedding_model = app_state["en_embedding_model"] if lang == 'en' else app_state["zh_embedding_model"]
            logging.info(f"Embedding query using '{lang}' model...")
            query_vector = embedding_model.encode([query_message])[0] # Synchronous call
            logging.info(f"Query embedding complete ({(time.time() - query_embedding_start_time):.2f}s).")
        except Exception as e:
            logging.error(f"Error embedding query: {e}")
            return jsonify({"detail": "Failed to embed query"}), 500 # Internal Server Error

        # --- 3. Search Vector Stores (Synchronous Search) ---
        logging.info("準備同步搜索向量庫...")
        retrieved_results = []
        search_start_time = time.time()
        # Search content stores
        for field in FIELDS_TO_VECTORIZE:
            key = f"{field}_{lang}"
            # ChromaDB Search
            chroma_collection = app_state.get("chroma_collections", {}).get(key)
            if chroma_collection:
                logging.debug(f"  查詢 Chroma Collection: {chroma_collection.name}")
                chroma_res = query_chroma_sync(chroma_collection, query_vector, TOP_K_CONTENT)
                if chroma_res:
                    ids,dists,metas,docs=chroma_res.get('ids',[[]])[0],chroma_res.get('distances',[[]])[0],chroma_res.get('metadatas',[[]])[0],chroma_res.get('documents',[[]])[0]
                    for j in range(len(ids)):
                        if 'source_field' not in metas[j]: metas[j]['source_field'] = field
                        retrieved_results.append({ "id": ids[j], "distance": dists[j], "metadata": metas[j], "document": docs[j], "source_db": "ChromaDB" })
            # FAISS Search
            faiss_index = app_state.get("faiss_indices", {}).get(key)
            faiss_meta = app_state.get("faiss_metadata", {}).get(key)
            if faiss_index and faiss_meta:
                logging.debug(f"  查詢 FAISS Index: {key}")
                faiss_res = query_faiss_sync(faiss_index, faiss_meta, query_vector, TOP_K_CONTENT)
                for res in faiss_res: res["source_db"] = "FAISS"; retrieved_results.append(res)

        # Search Message History Store
        history_collection = app_state.get("message_history_collection")
        if history_collection:
            logging.debug(f"  查詢 Message History Collection: {MESSAGE_HISTORY_COLLECTION_NAME}")
            hist_res = query_chroma_sync(history_collection, query_vector, TOP_K_HISTORY)
            if hist_res:
                ids,dists,metas,docs=hist_res.get('ids',[[]])[0],hist_res.get('distances',[[]])[0],hist_res.get('metadatas',[[]])[0],hist_res.get('documents',[[]])[0]
                for j in range(len(ids)):
                     if 'source_field' not in metas[j]: metas[j]['source_field'] = 'message_history'
                     retrieved_results.append({ "id": ids[j], "distance": dists[j], "metadata": metas[j], "document": docs[j], "source_db": "ChromaDB" })
        logging.info(f"Vector search complete ({(time.time() - search_start_time):.2f}s).")

        # --- 4. Rank/Sort combined results ---
        retrieved_results.sort(key=lambda x: x.get('distance', float('inf')))
        logging.info(f"Retrieved and sorted {len(retrieved_results)} results.")

        # --- 5. Format Context ---
        context_limit = TOP_K_CONTENT * len(FIELDS_TO_VECTORIZE) * 2 + TOP_K_HISTORY
        selected_results_for_context = retrieved_results[:context_limit]
        formatted_context = format_context(selected_results_for_context)
        logging.debug(f"Formatted Context (start):\n{formatted_context[:500]}...")
        # TODO: Implement token counting & truncation

        # --- 6. Construct LLM Prompt ---
        prompt = textwrap.dedent(f"""
            Based on the following retrieved context (including previous conversation turns if any, retrieved from FAISS and ChromaDB), please answer the user's question. If the context is empty or irrelevant, please state that you cannot answer based on the provided information.

            Context:
            {formatted_context if formatted_context else "No relevant context found."}

            User Question: {query_message}

            Answer:""")

        # --- 7. Call Target LLM (Synchronous) ---
        logging.info(f"Sending request to Groq LLM ({target_model} or fallback)...")
        llm_response = call_groq_llm_sync(app_state["groq_client"], prompt, target_model)
        if not llm_response:
            return jsonify({"detail": "Failed to get response from LLM"}), 500

        # --- 8. Update Message History (Background Thread) ---
        if history_collection:
            logging.info("Scheduling message history update in background thread...")
            update_thread = threading.Thread(
                target=update_history_background,
                args=(history_collection, query_message, llm_response),
                daemon=True # Allow main process to exit even if thread is running
            )
            update_thread.start()
        else:
            logging.warning("History collection unavailable, skipping update.")

        # --- 9. Return Response ---
        end_time = time.time()
        logging.info(f"Request processed successfully in {(end_time - request_start_time):.2f} seconds.")
        # Return JSON response with 200 OK status (implicit)
        return jsonify({"response": llm_response, "detected_language": lang})

    except Exception as e:
        # Catch any unexpected errors during request processing
        logging.exception(f"處理請求 '/rag_query' 時發生意外錯誤: {e}") # Log full traceback
        return jsonify({"detail": "An internal server error occurred"}), 500

# --- 7. Main Execution Block (for running Flask dev server) ---
if __name__ == "__main__":
    # Check if resources loaded successfully before starting server
    if not app_state.get("resources_ready", False):
         logging.critical("關鍵資源載入失敗，伺服器無法啟動。請檢查日誌。")
         exit(1) # Exit if loading failed

    logging.info("Starting Flask development server...")
    # Run Flask's built-in development server
    # Use threaded=True for basic concurrency handling
    app.run(host="127.0.0.1", port=SERVER_PORT, debug=True, use_reloader=False, threaded=True)