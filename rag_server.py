# -*- coding: utf-8 -*-
# ==============================================================================
# RAG Server using Flask (Synchronous with Background History Update)
# Incorporating query_content into the LLM prompt - No Semicolons
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
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
import textwrap
import threading
from flask_cors import CORS
# Import Tokenizer (optional but recommended for accurate truncation)
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    logging.warning("transformers library not found. Token counting and truncation will be approximate.")
    TOKENIZER_AVAILABLE = False


# Example extension ID, replace with your actual ID if needed for CORS
# extension_ID = 'plmphbheelmdnicgehmagnlhfahgjkme'
extension_ID = 'odmikollhnahmfohmdafkpnlfpopicmh'

# --- 1. Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
load_dotenv()
logging.info("環境變數已載入 (Environment variables loaded).")

# --- Paths and Model Names ---
EN_EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
ZH_EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
MODEL_FALLBACK_LIST = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
]
# Tokenizer for context truncation - choose one similar to your LLMs
TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

logging.info(f"訊息： 將依序嘗試以下 Groq Llama 模型進行生成: {MODEL_FALLBACK_LIST}")
FAISS_BASE_DIR = "faiss_w3c_stores"
CHROMA_BASE_DIR = "chroma_w3c_stores"
MESSAGE_HISTORY_COLLECTION_NAME = "conversation_history"
FIELDS_TO_VECTORIZE = [
    "abstract",
    "status_of_document",
    "content_summary",
    "original_content_snippet",
]
LANGUAGES = ["en", "zh"]
# Token Limit Configuration
MAX_PROMPT_TOKENS = 5500 # Max tokens for the entire prompt sent to LLM (leave buffer)
MAX_PROCESSED_CONTENT_TOKENS = 1000 # Max tokens for the summarized/extracted page content
# RAG retrieval parameters
TOP_K_CONTENT = 2
TOP_K_HISTORY = 3
SERVER_PORT = 5050

# Evaluation data filename
EVAL_DATA_FILENAME = "eval_data.json"

# --- Global Application State ---
app_state = {} # Dictionary to hold loaded resources

# --- 2. Pydantic Models for Request Validation ---
class RAGRequest(BaseModel):
    """Validates the incoming request body."""
    title: str
    query_message: str
    model_name: str
    page_content: str

# --- 3. Helper Functions ---
def get_faiss_paths(base_dir: str, field_name: str, lang: str) -> tuple[str, str]:
    """Generates standardized file paths for FAISS index and metadata."""
    filename_base = f"{field_name}_{lang}"
    index_path = os.path.join(base_dir, f"{filename_base}.index")
    metadata_path = os.path.join(base_dir, f"{filename_base}_meta.json")
    logging.debug(
        f"Generated FAISS paths: Index='{index_path}', Meta='{metadata_path}'"
    )
    return index_path, metadata_path

def get_chroma_collection_name(field_name: str, lang: str) -> str:
    """Generates a standardized collection name for ChromaDB."""
    collection_name = f"w3c_{field_name}_{lang}"
    logging.debug(f"Generated Chroma collection name: {collection_name}")
    return collection_name

def count_chars(text: str) -> tuple[int, int]:
    """Counts English letters and Chinese characters in a string."""
    en_count = len(re.findall(r"[a-zA-Z]", text))
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    return en_count, zh_count

def detect_language(text: str) -> str:
    """
    Detects language ('en' or 'zh') primarily using character counting,
    with langdetect as a fallback for ambiguous cases. Defaults to 'en'.
    """
    en_count, zh_count = count_chars(text)
    logging.debug(f"Character counts: EN={en_count}, ZH={zh_count}")

    if zh_count > 0 and zh_count >= en_count:
        final_lang = "zh"
    elif en_count > zh_count:
        final_lang = "en"
    else:
        logging.info("Character count analysis inconclusive, falling back to langdetect.")
        try:
            lang = detect(text)
            if lang.startswith("zh"):
                final_lang = "zh"
            elif lang == "en":
                final_lang = "en"
            else:
                final_lang = "en"  # Default to English for unsupported languages
            logging.debug(f"langdetect result: {lang}")
        except LangDetectException:
            logging.warning("Language detection by langdetect failed, defaulting to 'en'.")
            final_lang = "en"

    logging.info(f"Language determined: {final_lang}")
    return final_lang

def format_context(results: List[Dict]) -> str:
    """Formats retrieved search results into a single context string for LLM."""
    context_parts = []
    for res in results:
        metadata = res.get("metadata", {})
        doc_text = res.get("document", "")
        title = metadata.get("title", "N/A")
        source_field = metadata.get("source_field", "N/A")
        role = metadata.get("role")
        source_db = res.get("source_db", "Unknown")

        if role:
            prefix = f"--- History ({role}) [{source_db}] ---"
        else:
            prefix = f"--- Context from {source_field} (Title: {title}) [{source_db}] ---"
        # Use document text if available, else fallback to title (for FAISS results)
        content_to_add = doc_text if doc_text else f"Retrieved item with Title: {title}"
        context_parts.append(f"{prefix}\n{content_to_add}")
    # Join all formatted parts with double newlines
    return "\n\n".join(context_parts).strip()

# --- Synchronous LLM Call Function ---
def call_groq_llm_sync(client: Groq, prompt: str, target_model: str) -> Optional[str]:
    """Calls Groq LLM API synchronously with model fallback."""
    models_to_try = [target_model] + [
        m for m in MODEL_FALLBACK_LIST if m != target_model
    ]
    last_error = None
    for model_id in models_to_try:
        logging.debug(f"Attempting LLM generation with model: {model_id}")
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant answering questions based on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model_id,
                temperature=0.5,
                max_tokens=1024, # Limit LLM generation length
            )
            response = chat_completion.choices[0].message.content.strip()
            if response:
                logging.info(f"LLM success: {model_id}")
                return response
            else:
                logging.warning(f"{model_id} empty response.")
                last_error = "Empty response"
                continue
        except RateLimitError as e:
            logging.warning(f"Rate limit {model_id}. Trying next...")
            last_error = e
            time.sleep(1) # Wait before retrying
            continue
        except APIError as e_api:
            logging.error(f"API error {model_id}: {e_api}")
            last_error = e_api
            if e_api.status_code < 500:
                break # Stop on client errors
            else:
                time.sleep(1) # Wait before retry on server error
                continue
        except Exception as e:
            logging.error(f"Unexpected LLM error {model_id}: {e}")
            last_error = e
            break
    logging.error(f"LLM failed. Last error: {last_error}")
    return None

# --- Synchronous Query Functions ---
def query_chroma_sync(collection, query_vector, k):
    """Synchronously queries a ChromaDB collection."""
    try:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        return results
    except Exception as e_query:
        logging.error(f"Sync ChromaDB query failed for '{collection.name}': {e_query}")
        return None # Return None to indicate query failure

def query_faiss_sync(index, metadata_list, query_vector, k):
    """Synchronously searches a FAISS index."""
    try:
        query_vec_np = query_vector.reshape(1, -1).astype("float32")
        distances, indices = index.search(query_vec_np, k)
        results = []
        if indices.size > 0:
            for i in range(min(k, len(indices[0]))):
                idx = indices[0][i]
                dist = distances[0][i]
                if 0 <= idx < len(metadata_list):
                    meta = metadata_list[idx]
                    # Format FAISS result consistently
                    results.append(
                        {
                            "id": f"faiss_{idx}",
                            "distance": float(dist),
                            "metadata": meta,
                            "document": meta.get("title", "N/A") + " (FAISS)", # Placeholder
                        }
                    )
                else:
                    logging.warning(f"FAISS invalid idx:{idx}")
        return results
    except Exception as e_query:
        logging.error(f"Sync FAISS search failed: {e_query}")
        return [] # Return empty list on failure

# --- Function to update history in a background thread ---
def update_history_background(history_collection, query_message, llm_response):
    """Embeds and upserts query/response to ChromaDB history in a separate thread."""
    try:
        if "en_embedding_model" not in app_state or history_collection is None:
            logging.error("History Update: Required resources missing.")
            return
        history_embedding_model = app_state["en_embedding_model"]

        logging.info("Background History Update: Embedding...")
        # Perform blocking embedding in this background thread
        query_embedding_hist = history_embedding_model.encode([query_message])
        response_embedding_hist = history_embedding_model.encode([llm_response])
        # Safely convert embeddings to lists
        query_embed_list = (
            query_embedding_hist[0].tolist()
            if query_embedding_hist is not None and len(query_embedding_hist) > 0
            else None
        )
        response_embed_list = (
            response_embedding_hist[0].tolist()
            if response_embedding_hist is not None and len(response_embedding_hist) > 0
            else None
        )

        if query_embed_list and response_embed_list:
            timestamp = time.time()
            user_query_id = f"hist_user_{timestamp}"
            assistant_response_id = f"hist_asst_{timestamp}"
            # Perform blocking upsert in this background thread
            history_collection.upsert(
                ids=[user_query_id, assistant_response_id],
                embeddings=[query_embed_list, response_embed_list],
                documents=[query_message, llm_response],
                metadatas=[
                    {"role": "user", "timestamp": timestamp},
                    {"role": "assistant", "timestamp": timestamp},
                ],
            )
            logging.info("Background History Update: Update successful.")
        else:
            logging.error("Background History Update: Failed to generate embeddings.")
    except Exception as e:
        logging.error(f"Background History Update Error: {e}")

# --- Helper Function: Token-based Truncation ---
def truncate_text_by_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens using the provided tokenizer."""
    # Check if tokenizer is available (transformers installed and loaded)
    if not TOKENIZER_AVAILABLE or tokenizer is None:
        # Fallback to character-based truncation if tokenizer is unavailable
        char_limit = max_tokens * 4 # Approximate character limit
        if len(text) > char_limit:
            logging.warning(f"Truncating text based on characters ({char_limit}) due to missing tokenizer.")
            return text[:char_limit] + "..."
        else:
            return text # No truncation needed based on chars

    # Proceed with token-based truncation if tokenizer is available
    try:
        tokens = tokenizer.encode(text) # Encode text into token IDs
        if len(tokens) > max_tokens:
            # Truncate the list of token IDs
            truncated_tokens = tokens[:max_tokens]
            # Decode the truncated token IDs back into a string
            truncated_text = tokenizer.decode(
                truncated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            logging.warning(f"Truncated text from {len(tokens)} to {max_tokens} tokens.")
            return truncated_text + "..." # Add ellipsis to indicate truncation
        else:
            return text # No truncation needed based on tokens
    except Exception as e:
        # Fallback to character limit if tokenization/decoding fails
        logging.error(f"Error during token truncation: {e}. Falling back to character limit.")
        char_limit = max_tokens * 4
        if len(text) > char_limit:
            return text[:char_limit] + "..."
        else:
            return text

# --- Helper Function: Preprocess Page Content ---
def preprocess_page_content_with_llm(client: Groq, query_message: str, page_content: str, target_model: str) -> str:
    """
    Uses LLM to extract relevant parts or summarize page_content based on query_message.
    Limits input and output size. Returns processed content or empty string.
    """
    # Limit input to LLM for preprocessing to avoid excessive tokens/cost
    max_preprocess_len = 6000 # Limit input content length for this step
    content_snippet = page_content[:max_preprocess_len] + ("..." if len(page_content) > max_preprocess_len else "")

    # Define prompt for the preprocessing LLM call
    prompt = textwrap.dedent(f"""
        User Query: "{query_message}"

        Page Content Snippet:
        ---
        {content_snippet}
        ---

        Based *only* on the User Query and the Page Content Snippet provided above, please perform the following task:
        1. Identify and extract the sentences or short paragraphs from the Page Content Snippet that are **most relevant** to answering the User Query.
        2. If relevant sections are found, present them concisely.
        3. If no truly relevant information is found in the snippet regarding the query, output the exact phrase: NO_RELEVANT_CONTENT_FOUND

        Output only the extracted relevant text or the specific phrase "NO_RELEVANT_CONTENT_FOUND". Do not add explanations or introductions.
    """)

    logging.info(f"Preprocessing page content based on query: '{query_message[:50]}...'")
    # Use a potentially faster/cheaper model for preprocessing
    preprocessing_model = "llama-3.1-8b-instant" if "llama-3.1-8b-instant" in MODEL_FALLBACK_LIST else target_model
    # Call the synchronous LLM helper function
    processed_content = call_groq_llm_sync(client, prompt, preprocessing_model)

    # Check if preprocessing was successful and relevant content was found
    if processed_content and "NO_RELEVANT_CONTENT_FOUND" not in processed_content:
        logging.info("Page content preprocessing successful.")
        # Truncate the *result* of preprocessing based on tokens before returning
        return truncate_text_by_tokens(
            processed_content, app_state.get("tokenizer"), MAX_PROCESSED_CONTENT_TOKENS
        )
    else:
        # Handle cases where preprocessing failed or found nothing relevant
        logging.warning("Page content preprocessing failed or found no relevant content.")
        return "" # Return empty string
    
def log_eval_data(title: str, query: str, model_name: str, generated_response: str):
    """將互動資訊記錄到 .json 檔案中，作為潛在的評估資料。"""
    try:
        entry = {
            "id": title,
            "query_message": query,
            "model_name": model_name,
            "generated_answer": generated_response,
            "expected_answer": "",
        }

        # Load existing eval data if the file exists
        if os.path.exists(EVAL_DATA_FILENAME):
            with open(EVAL_DATA_FILENAME, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        else:
            eval_data = {"eval_datas": []}

        # Append the new entry to the eval_datas list
        eval_data["eval_datas"].append(entry)

        # Write the updated eval data back to the file
        with open(EVAL_DATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=4)

        logging.debug(f"已將潛在評估資料記錄到 {EVAL_DATA_FILENAME}")

    except Exception as e:
        logging.error(f"記錄潛在評估資料時出錯: {e}")

# --- 4. Global Resource Loading (Synchronous) ---
logging.info("腳本啟動：正在全局同步載入資源...")
resources_loaded = True
start_load_time = time.time()

# 4.1 Groq Client
logging.info("  (1/5) 正在初始化 Groq Client...")
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key: raise ValueError("未在環境變數中找到 GROQ_API_KEY")
    app_state["groq_client"] = Groq(api_key=groq_api_key)
    logging.info("  (1/5) Groq Client 初始化成功。")
except Exception as e: logging.critical(f"  (1/5) 初始化 Groq Client 失敗: {e}"); resources_loaded = False

# 4.2 Embedding Models
if resources_loaded:
    logging.info("  (2/5) 正在載入 Embedding Models...")
    try:
        logging.info(f"    正在載入 EN Model: {EN_EMBEDDING_MODEL_NAME}")
        app_state["en_embedding_model"] = SentenceTransformer(EN_EMBEDDING_MODEL_NAME)
        logging.info(f"    EN Model 載入完成。")
        logging.info(f"    正在載入 ZH Model: {ZH_EMBEDDING_MODEL_NAME}")
        app_state["zh_embedding_model"] = SentenceTransformer(ZH_EMBEDDING_MODEL_NAME)
        logging.info("  (2/5) Embedding Models 載入成功。")
    except Exception as e: logging.critical(f"  (2/5) 載入 Embedding Models 失敗: {e}"); resources_loaded = False

# 4.3 Tokenizer
if resources_loaded and TOKENIZER_AVAILABLE:
    logging.info(f"  (3/5) 正在載入 Tokenizer ({TOKENIZER_NAME})...")
    try:
        # Load tokenizer synchronously
        app_state["tokenizer"] = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        logging.info("  (3/5) Tokenizer 載入成功。")
    except Exception as e:
        logging.error(f"  (3/5) 載入 Tokenizer 失敗: {e}. Context truncation will be approximate.")
        app_state["tokenizer"] = None
elif resources_loaded: # Tokenizer library not installed
     logging.warning("  (3/5) `transformers` 庫未安裝，跳過 Tokenizer 載入。")
     app_state["tokenizer"] = None
else: # Previous steps failed
     logging.warning("  (3/5) Skipping Tokenizer loading due to previous errors.")
     app_state["tokenizer"] = None


# 4.4 ChromaDB Client and Collections (Now step 4/5)
if resources_loaded:
    logging.info(f"  (4/5) 正在初始化 ChromaDB Client (路徑: {CHROMA_BASE_DIR})...")
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
        logging.info(f"  (4/5) ChromaDB Collections 處理完成 ({loaded_chroma_count} 個成功)。")
    except Exception as e: logging.critical(f"  (4/5) 初始化 ChromaDB 失敗: {e}"); resources_loaded = False

# 4.5 Load FAISS Indices and Metadata (Now step 5/5)
if resources_loaded:
    logging.info(f"  (5/5) 正在載入 FAISS 索引和元資料 (來自: {FAISS_BASE_DIR})...")
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
    except Exception as e: logging.critical(f"  (5/5) 載入 FAISS 資料時發生嚴重錯誤: {e}"); resources_loaded = False
    logging.info(f"  (5/5) FAISS 索引和元資料載入完成 ({faiss_loaded_count} 個成功)。")

# Final Check and Log
app_state["resources_ready"] = resources_loaded
if resources_loaded:
     end_load_time = time.time()
     logging.info(f"--- 所有資源全局載入完成，總耗時: {(end_load_time - start_load_time):.2f} 秒。---")
else:
     logging.critical("--- 資源載入失敗，伺服器可能無法正常工作。 ---")

# --- 5. Flask App Initialization ---
app = Flask(__name__)
logging.info("Flask 應用實例已創建。")
# Enable CORS for the specific extension ID
CORS(app, origins=f"chrome-extension://{extension_ID}", supports_credentials=True)
logging.info(f"CORS enabled for origin: chrome-extension://{extension_ID}")

# --- 6. API Endpoint Definition ---
@app.route("/rag_query", methods=["POST"])
def perform_rag_flask():
    """API endpoint using Flask to handle RAG requests."""
    request_start_time = time.time()
    try:
        logging.info(f"--- Endpoint Start (/rag_query) ---")
        resources_ready_flag = app_state.get("resources_ready", False)
        logging.info(f"Endpoint check: resources_ready = {resources_ready_flag}")
        logging.debug(f"Current app_state keys: {list(app_state.keys())}")

        # --- Dependency Check ---
        if not resources_ready_flag:
            logging.error("伺服器資源未就緒。拒絕請求。")
            return jsonify({"detail": "Server resources not ready"}), 503

        # Validate request data
        try:
            request_json = request.get_json()
            if not request_json:
                 return jsonify({"detail": "Request body must be JSON"}), 400
            # Validate using the updated Pydantic model
            request_data = RAGRequest.model_validate(request_json)
        except ValidationError as e:
             logging.error(f"請求體驗證失敗: {e}")
             return jsonify({"detail": e.errors()}), 400
        except Exception as e:
             logging.error(f"解析請求 JSON 或驗證時出錯: {e}")
             return jsonify({"detail": "Invalid JSON request body or validation error"}), 400

        title = request_data.title
        query_message = request_data.query_message
        target_model = request_data.model_name
        original_page_content = request_data.page_content # Get original page content
        logging.info(
            f"收到請求: query='{query_message[:50]}...', model='{target_model}', content provided: {len(original_page_content)>0}"
        )

        # --- 1. Detect Language ---
        lang = detect_language(query_message)
        logging.info(f"Detected language: {lang}")

        # --- 2. Embed Query ---
        query_embedding_start_time = time.time()
        try:
            embedding_model = (
                app_state["en_embedding_model"]
                if lang == "en"
                else app_state["zh_embedding_model"]
            )
            logging.info(f"Embedding query using '{lang}' model...")
            query_vector = embedding_model.encode([f"title：{title}\nquestion：{query_message}"])[0]
            logging.info(
                f"Query embedding complete ({(time.time() - query_embedding_start_time):.2f}s)."
            )
        except Exception as e:
            logging.error(f"Error embedding query: {e}")
            return jsonify({"detail": "Failed to embed query"}), 500

        # --- 3. Search Vector Stores (Synchronous) ---
        logging.info("準備同步搜索向量庫...")
        retrieved_results = []
        search_start_time = time.time()
        # Search content stores
        for field in FIELDS_TO_VECTORIZE:
            key = f"{field}_{lang}"
            chroma_collection = app_state.get("chroma_collections", {}).get(key)
            if chroma_collection:
                logging.debug(f"  查詢 Chroma Collection: {chroma_collection.name}")
                chroma_res = query_chroma_sync(chroma_collection, query_vector, TOP_K_CONTENT)
                if chroma_res:
                    ids,dists,metas,docs=chroma_res.get('ids',[[]])[0],chroma_res.get('distances',[[]])[0],chroma_res.get('metadatas',[[]])[0],chroma_res.get('documents',[[]])[0]
                    for j in range(len(ids)): metas[j]['source_field']=field; retrieved_results.append({"id":ids[j],"distance":dists[j],"metadata":metas[j],"document":docs[j],"source_db":"ChromaDB"})
            faiss_index = app_state.get("faiss_indices", {}).get(key)
            faiss_meta = app_state.get("faiss_metadata", {}).get(key)
            if faiss_index and faiss_meta:
                logging.debug(f"  查詢 FAISS Index: {key}")
                faiss_res = query_faiss_sync(faiss_index, faiss_meta, query_vector, TOP_K_CONTENT)
                for res in faiss_res: res["source_db"]="FAISS"; retrieved_results.append(res)
        # Search history store
        history_collection = app_state.get("message_history_collection")
        if history_collection:
            logging.debug(f"  查詢 Message History: {MESSAGE_HISTORY_COLLECTION_NAME}")
            hist_res = query_chroma_sync(history_collection, query_vector, TOP_K_HISTORY)
            if hist_res:
                ids,dists,metas,docs=hist_res.get('ids',[[]])[0],hist_res.get('distances',[[]])[0],hist_res.get('metadatas',[[]])[0],hist_res.get('documents',[[]])[0]
                for j in range(len(ids)): metas[j]['source_field']='message_history'; retrieved_results.append({"id":ids[j],"distance":dists[j],"metadata":metas[j],"document":docs[j],"source_db":"ChromaDB"})
        logging.info(f"Vector search complete ({(time.time() - search_start_time):.2f}s).")

        # --- 4. Rank/Sort combined results ---
        retrieved_results.sort(key=lambda x: x.get("distance", float("inf")))
        logging.info(f"Retrieved and sorted {len(retrieved_results)} results.")

        # --- 5. Format Retrieved Context AND Preprocess Page Content ---
        # Format the context retrieved from vector stores
        context_limit = TOP_K_CONTENT * len(FIELDS_TO_VECTORIZE) * 2 + TOP_K_HISTORY
        selected_results_for_context = retrieved_results[:context_limit]
        formatted_retrieved_context = format_context(selected_results_for_context)
        logging.debug(f"Formatted Retrieved Context (start):\n{formatted_retrieved_context[:500]}...")

        # Preprocess the original page content using LLM
        processed_page_content = preprocess_page_content_with_llm(
            app_state["groq_client"], query_message, original_page_content, target_model
        )
        logging.debug(f"Processed Page Content (start):\n{processed_page_content[:500]}...")

        # --- 6. Construct Final LLM Prompt (using both contexts) ---
        prompt = textwrap.dedent(f"""
            You are a helpful AI assistant knowledgeable about W3C standards. This question you should answer in {"Englsih" if lang == "en" else "Traditional Chinese"}.
            Carefully analyze BOTH the Retrieved Context from internal knowledge AND the relevant snippet from the User's Current Page Content.
            Provide a comprehensive and accurate answer based *only* on the provided information (both retrieved context and page content) and the user's question.
            If the necessary information isn't found in either source, state that clearly. Prioritize information from the Retrieved Context if there's overlap but mention if the page content offers additional relevant details.
            Do not make up information. Structure your answer clearly.

            **Retrieved Context (from Vector Stores):**
            ---
            {formatted_retrieved_context if formatted_retrieved_context else "No relevant context was found from vector stores."}
            ---

            **Relevant Snippet from User's Current Page Content:**
            ---
            {processed_page_content if processed_page_content else "No current page content provided or processed."}
            ---

            **User's Question:**
            {query_message}

            **Answer:**
        """)
        logging.debug(f"Final Prompt for LLM (start):\n{prompt[:600]}...")

        # --- 7. Call Target LLM (Synchronous) ---
        logging.info(f"Sending request to Groq LLM ({target_model} or fallback)...")
        llm_response = call_groq_llm_sync(
            app_state["groq_client"], prompt, target_model
        )

        if llm_response:
            log_eval_data(
                title=title,
                query=query_message,
                model_name=target_model,
                generated_response=llm_response
            )
        elif not llm_response:
            return jsonify({"detail": "Failed to get response from LLM"}), 500

        # --- 8. Update Message History (Background Thread) ---
        if history_collection:
            logging.info("Scheduling message history update...")
            update_thread = threading.Thread(
                target=update_history_background,
                args=(history_collection, query_message, llm_response),
                daemon=True,
            )
            update_thread.start()
        else:
            logging.warning("History collection unavailable.")

        # --- 9. Return Response ---
        end_time = time.time()
        logging.info(
            f"Request processed successfully in {(end_time - request_start_time):.2f} seconds."
        )
        return jsonify({"response": llm_response, "detected_language": lang})

    except Exception as e:
        logging.exception(f"處理請求 '/rag_query' 時發生意外錯誤: {e}")
        return jsonify({"detail": "An internal server error occurred"}), 500

@app.route("/clear_memory", methods=["POST"])
def clear_conversation_history():
    """API endpoint to clear the conversation history."""
    logging.info("--- Endpoint Start (/clear_memory) ---")
    try:
        chroma_client = app_state.get("chroma_client")
        history_ef = app_state.get("chroma_ef_map", {}).get('en')

        if not chroma_client:
            logging.error("ChromaDB client not available.")
            return jsonify({"detail": "ChromaDB client not initialized"}), 500
        if not history_ef:
            logging.error("History embedding function not available.")
            return jsonify({"detail": "History embedding function not initialized"}), 500

        logging.info(f"Attempting to clear history collection: {MESSAGE_HISTORY_COLLECTION_NAME}")

        # Delete the existing collection
        try:
            chroma_client.delete_collection(name=MESSAGE_HISTORY_COLLECTION_NAME)
            logging.info(f"Collection '{MESSAGE_HISTORY_COLLECTION_NAME}' deleted.")
        except Exception as e_del:
            # Log the error but proceed, as get_or_create_collection will handle creation
            logging.warning(f"Could not delete collection '{MESSAGE_HISTORY_COLLECTION_NAME}' (might not exist): {e_del}")
            # You might want to check the specific exception type here if needed

        # Recreate the collection (empty)
        try:
            new_history_collection = chroma_client.get_or_create_collection(
                name=MESSAGE_HISTORY_COLLECTION_NAME,
                embedding_function=history_ef,
                metadata={"hnsw:space": "l2"} # Ensure settings match initial creation
            )
            # Update the reference in app_state
            app_state["message_history_collection"] = new_history_collection
            logging.info(f"Collection '{MESSAGE_HISTORY_COLLECTION_NAME}' recreated successfully.")
            return jsonify({"status": "success", "message": "Conversation history cleared."}), 200
        except Exception as e_create:
            logging.error(f"Failed to recreate history collection '{MESSAGE_HISTORY_COLLECTION_NAME}': {e_create}")
            # Attempt to set the app_state reference to None if creation fails
            app_state["message_history_collection"] = None
            return jsonify({"detail": f"Failed to recreate history collection: {e_create}"}), 500

    except Exception as e:
        logging.exception(f"處理請求 '/clear_memory' 時發生意外錯誤: {e}")
        return jsonify({"detail": "An internal server error occurred while clearing history"}), 500

# --- 7. Main Execution Block (for running Flask dev server) ---
if __name__ == "__main__":
    if not app_state.get("resources_ready", False):
        logging.critical("關鍵資源載入失敗，伺服器無法啟動。")
        exit(1)

    logging.info(f"Starting Flask development server on http://127.0.0.1:{SERVER_PORT}...")
    # Run Flask dev server
    app.run(
        host="127.0.0.1",
        port=SERVER_PORT,
        debug=True,         # Enable reloader and debugger
        use_reloader=True, # Explicitly enable reloader for Flask dev server
        threaded=True      # Handle requests in threads
    )