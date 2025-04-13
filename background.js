const RAG_SERVER_URL = "http://127.0.0.1:5050/rag_query";
const DEFAULT_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "queryMessage") {
        const userQuestion = request.user_question;
        const modelName = request.model_name || DEFAULT_MODEL_NAME;

        if (!userQuestion) {
            console.error("❌ 未提供 user_question");
            sendResponse({ status: "error", response: "請提供 user_question。" });
            return false;
        }

        // 從 chrome.storage.local 中取得頁面內容
        chrome.storage.local.get(["w3cContent", "articleTitle"], (result) => {
            const pageContent = result.w3cContent;
            const pageTitle = result.articleTitle;

            if (!pageContent) {
                console.error("❌ 尚未儲存頁面內容 (w3cContent)");
                sendResponse({ status: "error", response: "請先提取當前頁面的內容。" });
                return;
            }
            if (!pageTitle) {
                console.error("❌ 尚未儲存頁面標題 (articleTitle)");
                sendResponse({ status: "error", response: "請先提取當前頁面的標題。" });
                return;
            }

            // 建立 query_message 格式
            // `這是使用者正在閱讀的文章：\n${pageContent}\n\n使用者提問：${userQuestion}`
            const queryMessage = `優先搜尋：${pageTitle}\n使用者提問：${userQuestion}`;

            console.log("📨 傳送 query_message 至伺服器...");
            console.log("🧠 模型名稱:", modelName);

            // 發送給後端
            fetch(RAG_SERVER_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query_message: queryMessage,
                    model_name: modelName,
                    page_content: pageContent,
                })
            })
            .then(async response => {
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    const errorMessage = errorData.error || `HTTP 錯誤 ${response.status}`;
                    throw new Error(`伺服器錯誤：${errorMessage}`);
                }
                return response.json();
            })
            .then(data => {
                const reply = data.response || "AI 沒有提供任何回答。";
                console.log("✅ 收到伺服器回應。");
                sendResponse({ status: "success", response: reply });
            })
            .catch(error => {
                console.error("❌ 發送查詢時發生錯誤:", error);
                sendResponse({ status: "error", response: `錯誤：${error.message}` });
            });
        });

        return true; // 非同步回應
    }
});

console.log("🏁 Background script 已載入（使用 storage 中的頁面內容 + user_question）");
