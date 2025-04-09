let extractedContent = "";
const GEMINI_API_KEY = "AIzaSyA0HMdHi6ceZSvb7f60weMqEDu8easuui0";
const GEMINI_MODEL = 'gemini-1.5-flash-latest';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("Background received message:", request.action);

    // --- 處理文本提取 ---
    if (request.action === "extractText") {
        extractedContent = request.content;
        console.log("✅ 已存儲 W3C 內容，字數:", extractedContent.length);
        sendResponse({ status: "success", message: "內容提取成功。" });
        return false;
    }

    // --- 處理 Gemini 查詢 ---
    if (request.action === "queryGemini") {
        chrome.storage.local.get("w3cContent", (result) => {
            const extractedContent = result.w3cContent;
            
            if (!extractedContent) {
                sendResponse({ status: "no_content", message: "請先從 W3C 頁面提取內容。" });
                return;
            }
        
            const prompt = `請根據以下 W3C 標準文件的內容，回答使用者的問題。如果適合，請使用 Markdown 格式來組織你的回答。
        
        標準內容:
        ---
        ${extractedContent}
        ---
        
        使用者問題: ${request.question}
        
        回答:`;
        
            const requestBody = { contents: [{ parts: [{ text: prompt }] }] };
        
            fetch(GEMINI_API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            })
            .then(async response => {
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    const errorMessage = errorData?.error?.message || response.statusText;
                    throw new Error(`API 錯誤 ${response.status}: ${errorMessage}`);
                }
                return response.json();
            })
            .then(data => {
                let aiResponse = "AI 未提供有效回應。";
                if (data?.candidates?.[0]?.content?.parts?.[0]?.text) {
                    aiResponse = data.candidates[0].content.parts[0].text;
                } else if (data?.promptFeedback?.blockReason) {
                    aiResponse = `請求被 API 阻止：${data.promptFeedback.blockReason}`;
                    console.warn("Gemini 請求被阻止:", data.promptFeedback);
                }
                sendResponse({ status: "success", response: aiResponse });
            })
            .catch(error => {
                console.error("❌ Gemini API 錯誤:", error);
                sendResponse({ status: "error", response: `AI 連線失敗: ${error.message}` });
            });
        });

        return true; // 異步響應
    }

    // ** 移除 renderMarkdown 處理邏輯 **

});

console.log("🏁 Background script loaded.");