import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

    if zh_count > 0 and zh_count >= en_count:
        final_lang = "zh"
    else:
        final_lang = "en"
    return final_lang

# 模型選擇（根據語言）
en_model = SentenceTransformer("all-mpnet-base-v2")
zh_model = SentenceTransformer("shibing624/text2vec-base-chinese")

with open("eval_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

generated_answers = [item["generated_answer"] for item in data["eval_datas"]]
expected_answers = [item["expected_answer"] for item in data["eval_datas"]]

for i, item in enumerate(data["eval_datas"]):
    # 根據語言選擇模型
    if detect_language(item["generated_answer"]) == "zh":
        model = zh_model
    else:
        model = en_model
    
    # 編碼生成答案和標準答案
    gen_emb = model.encode([generated_answers[i]])[0]  # 擷取一維向量
    exp_emb = model.encode([expected_answers[i]])[0]  # 擷取一維向量

    # 計算語意相似度
    similarity_score = cosine_similarity([gen_emb], [exp_emb])[0][0]
    print(f"Generated Answer: {generated_answers[i]}")
    print(f"Expected Answer: {expected_answers[i]}")
    print(f"Cosine Similarity: {similarity_score:.4f}\n")
