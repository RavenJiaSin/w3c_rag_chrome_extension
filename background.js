let extractedContent = ""; // 存儲 W3C 內容

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extractText") {
        extractedContent = request.content; // 更新存儲的標準內容
        console.log("✅ 已存儲 W3C 內容，字數:", extractedContent.length);
        sendResponse({ status: "success" });
    }
    if (request.action === "queryOllama") {
        if (!extractedContent) {
            sendResponse({ response: "❌ 尚未擷取標準內容，請先載入 W3C 頁面" });
            return;
        }

        console.log("📡 正在向 Ollama 發送請求...", request.question);
        fetch("http://localhost:11434/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: "llama3",
                prompt: `這是一份 W3C 標準文件的一部分，請根據這些內容回答問題。\n\n標準內容:\n${extractedContent}\n\n使用者問題: ${request.question}`,
                stream: false
            })
        })
            .then(response => response.json())
            .then(data => {
                console.log("✅ API 回應:", data);
                sendResponse({ response: data.response || "AI 沒有回應" });
            })
            .catch(error => {
                console.error("❌ Ollama 錯誤:", error);
                sendResponse({ response: "AI 連線失敗" });
            });

        return true; // 讓 sendResponse 可用於異步回應
    }

});
