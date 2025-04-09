let extractedContent = ""; // 確保這個變數仍然用於存儲頁面內容
const GEMINI_API_KEY = "AIzaSyA0HMdHi6ceZSvb7f60weMqEDu8easuui0"; // 考慮更安全地儲存金鑰
const GEMINI_MODEL = 'gemini-1.5-flash-latest';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;
const RAG_SERVER_URL = "http://127.0.0.1:5000/search"; // 你的 Python Flask 伺服器的 URL

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("後台收到訊息:", request.action);

    // --- 處理文本提取 ---
    if (request.action === "extractText") {
        extractedContent = request.content; // 儲存原始提取的內容
        console.log("✅ 已儲存當前頁面 W3C 內容，長度:", extractedContent.length);
        sendResponse({ status: "success", message: "當前頁面內容提取成功。" });
        return false; // 同步回應
    }

    // --- 處理結合頁面內容和 RAG 的查詢 ---
    if (request.action === "queryGemini") {
        const userQuestion = request.question; // 從請求中獲取使用者問題
        if (!userQuestion) {
             console.error("錯誤：未提供問題。");
             sendResponse({ status: "error", response: "請提供一個問題。" });
             return false;
        }
        if (!GEMINI_API_KEY) {
             console.error("錯誤：缺少或無效的 Gemini API 金鑰。");
             sendResponse({ status: "error", response: "缺少或無效的 Gemini API 金鑰。" });
             return false;
         }

        chrome.storage.local.get("w3cContent", (result) => {
            const extractedContent = result.w3cContent;

            // --- *** 新增檢查：確保頁面內容已被提取 *** ---
            if (!extractedContent) {
                console.error("錯誤：尚未提取當前頁面內容。無法執行結合查詢。");
                sendResponse({ status: "error", response: "請先提取當前頁面的內容，然後再提問。" });
                return false; // 阻止繼續執行
            }
            // --- *** 檢查結束 *** ---

            console.log(`❓ 使用者問題: ${userQuestion}`);
            console.log(`📄 使用當前頁面內容 (長度: ${extractedContent.length})`);
            console.log(`📡 正在向 RAG 伺服器 ${RAG_SERVER_URL} 查詢相關輔助上下文...`);

            // --- 步驟 1: 從 RAG 伺服器獲取輔助上下文 ---
            fetch(RAG_SERVER_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: userQuestion })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().catch(() => ({})).then(errData => {
                        const errMsg = errData.error || `HTTP 錯誤 ${response.status}`;
                        throw new Error(`RAG 伺服器錯誤: ${errMsg}`);
                    });
                }
                return response.json();
            })
            .then(ragData => {
                // 從回應中獲取檢索到的輔助上下文
                let retrievedContext = ragData.context;
                if (!retrievedContext) {
                    console.warn("RAG 伺服器未找到相關的輔助上下文。");
                    retrievedContext = "在相關標準資料庫中未找到額外的輔助資訊。"; // 提供一個預設值
                } else {
                    console.log("📚 已從 RAG 伺服器接收到輔助上下文。");
                }

                // --- 步驟 2: 建構結合兩者的提示 ---
                console.log("✨ 正在使用頁面內容和 RAG 輔助上下文為 Gemini 建構提示...");

                // ***** 修改後的 Prompt *****
                const prompt = `請根據以下提供的「當前頁面內容」和「相關輔助上下文」來回答使用者的問題。

                                **重要指示:**
                                1.  **你的回答必須主要基於「當前頁面內容」。**
                                2.  只有當「當前頁面內容」無法完整回答問題，或者需要補充細節/定義時，才參考「相關輔助上下文」。
                                3.  不要將「相關輔助上下文」的內容作為主要回答來源，它僅用於輔助理解和補充。
                                4.  如果兩個來源都無法提供答案，請明確說明資訊不存在於提供的內容中。
                                5.  如果適合，請使用 Markdown 格式化回答。

                                ---
                                **當前頁面內容 (主要依據):**
                                \`\`\`
                                ${extractedContent}
                                \`\`\`
                                ---
                                **相關輔助上下文 (僅供參考和補充):**
                                \`\`\`
                                ${retrievedContext}
                                \`\`\`
                                ---

                                **使用者問題:** ${userQuestion}

                                **回答:**`;
                // ***** Prompt 修改結束 *****

                // 準備發送給 Gemini API 的請求體
                const requestBody = { contents: [{ parts: [{ text: prompt }] }] };

                console.log("📡 正在向 Gemini API 發送請求 (結合上下文)...");
                // 返回 fetch Gemini API 的 Promise
                return fetch(GEMINI_API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestBody)
                });
            })
            .then(async geminiResponse => {
                // (後續處理 Gemini 回應的部分保持不變)
                if (!geminiResponse.ok) {
                    const errorData = await geminiResponse.json().catch(() => ({}));
                    const errorMessage = errorData?.error?.message || geminiResponse.statusText;
                    throw new Error(`Gemini API 錯誤 ${geminiResponse.status}: ${errorMessage}`);
                }
                return geminiResponse.json();
            })
            .then(geminiData => {
                // (提取和發送回應的部分保持不變)
                let aiResponse = "AI 未提供有效的回應。";
                if (geminiData?.candidates?.[0]?.content?.parts?.[0]?.text) {
                    aiResponse = geminiData.candidates[0].content.parts[0].text;
                } else if (geminiData?.promptFeedback?.blockReason) {
                    aiResponse = `請求被 API 阻止：${geminiData.promptFeedback.blockReason}`;
                    console.warn("Gemini 請求被阻止:", geminiData.promptFeedback);
                }
                console.log("✅ 已收到 Gemini API 回應。");
                sendResponse({ status: "success", response: aiResponse });
            })
            .catch(error => {
                // (錯誤處理部分保持不變)
                console.error("❌ 查詢過程中發生錯誤:", error);
                let userErrorMessage = `發生錯誤: ${error.message}`;
                if (error.message && (error.message.includes("Failed to fetch") || error.message.includes("NetworkError"))) {
                    userErrorMessage = `無法連接到本地 RAG 伺服器 (${RAG_SERVER_URL})。請確認 Python 伺服器 (server.py) 正在運行中。`;
                } else if (error.message && error.message.includes("RAG 伺服器錯誤")) {
                    userErrorMessage = `檢索輔助上下文時出錯: ${error.message}`;
                } else if (error.message && error.message.includes("Gemini API 錯誤")) {
                    userErrorMessage = `查詢 AI 時出錯: ${error.message}`;
                }
                sendResponse({ status: "error", response: userErrorMessage });
            });
        });
        
        // 必須返回 true，因為 fetch 是異步操作
        return true;
    }

});

console.log("🏁 Background script (支援結合頁面與 RAG) 已載入。");