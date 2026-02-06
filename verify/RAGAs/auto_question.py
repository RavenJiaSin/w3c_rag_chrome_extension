# -*- coding: utf-8 -*-
import json
import requests
import time

# --- 配置 ---
SERVER_URL = "http://127.0.0.1:5050/rag_query"  # Flask server endpoint
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # 你想用的模型
JSON_PATH = r"verify\RAGAs\RAGAs_data_en.json"
DELAY_BETWEEN_REQUESTS = 1  # 秒數，可避免 server 過載

# --- 讀取 JSON ---
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- 逐筆送出 user_input 並取得 response ---
for idx, entry in enumerate(data):
    user_input = entry.get("user_input", "").strip()
    if not user_input:
        print(f"[{idx+1}/{len(data)}] 空的 user_input，跳過")
        continue

    payload = {
        "title": f"RAGA Entry {idx+1}",
        "query_message": user_input,
        "model_name": MODEL_NAME,
        "page_content": ""  # 若無網頁內容可留空
    }

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        entry["response"] = result.get("response", "")
        print(f"[{idx+1}/{len(data)}] 完成: {user_input[:50]} -> {entry['response'][:50]}")

    except Exception as e:
        print(f"[{idx+1}/{len(data)}] 發生錯誤: {user_input[:50]}, {e}")
        entry["response"] = ""

    # 可選延遲，避免過快呼叫 server
    time.sleep(DELAY_BETWEEN_REQUESTS)

    # break # --- 測試用，先處理一筆就停，正式使用時請移除這行 ---

# --- 寫回 JSON ---
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("全部完成！")
