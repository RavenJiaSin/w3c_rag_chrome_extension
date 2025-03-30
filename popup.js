document.addEventListener("DOMContentLoaded", function () {
    const inputField = document.getElementById("question-input");
    const sendButton = document.getElementById("send-button");
    const responseDiv = document.getElementById("response");

    sendButton.addEventListener("click", function () {
        const userQuestion = inputField.value.trim();
        if (!userQuestion) {
            responseDiv.innerHTML = "<span style='color: orange;'>⚠️ 請先輸入問題。</span>";
            return;
        }

        // 清空上次的回應並顯示等待訊息
        responseDiv.innerHTML = "<span style='color: gray;'>📡 正在向 Gemini 發送請求...</span>";
        sendButton.disabled = true;
        inputField.disabled = true;

        chrome.runtime.sendMessage({ action: "queryGemini", question: userQuestion }, (response) => {
             sendButton.disabled = false;
             inputField.disabled = false;

            if (chrome.runtime.lastError) {
                console.error("擴充功能通訊錯誤:", chrome.runtime.lastError.message);
                responseDiv.innerHTML = "<span style='color: red;'>❌ 擴充功能錯誤：" + chrome.runtime.lastError.message + "</span>";
                return;
            }

            if (response) {
                 switch (response.status) {
                    case "success":
                        console.log("AI 回應 (原始):", response.response);
                        try {
                            // 1. 使用 marked 將 Markdown 解析為 HTML
                            const rawHtml = marked.parse(response.response || "");

                            // 2. 使用 DOMPurify 清理 HTML，防止 XSS
                            const cleanHtml = DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true }, ADD_ATTR: ['target'] });

                            // 3. 將清理後的 HTML 設置到 responseDiv
                            responseDiv.innerHTML = cleanHtml; // 直接顯示渲染後的 HTML

                        } catch (parseError) {
                             console.error("Markdown 解析或清理錯誤:", parseError);
                             // 回退方案：顯示原始文本，但保留換行
                             responseDiv.innerText = "⚠️ 無法渲染 Markdown，顯示原始文字:\n" + response.response;
                        }
                        break;
                    case "no_content":
                        console.log("無內容提示");
                        responseDiv.innerHTML = "<span style='color: orange;'>🟡 " + response.message + "</span>";
                        break;
                    case "error":
                        console.error("API 或內部錯誤:", response.response);
                        responseDiv.innerHTML = "<span style='color: red;'>❌ 錯誤：" + response.response + "</span>";
                        break;
                    default:
                        console.warn("收到未知的回應狀態:", response);
                        responseDiv.innerHTML = "<span style='color: orange;'>❓ 收到未知的回應。</span>";
                 }
            } else {
                 console.error("收到無效的回應 (undefined/null)");
                 responseDiv.innerHTML = "<span style='color: red;'>❌ 收到來自背景腳本的無效回應。</span>";
            }
        });
    });

    inputField.addEventListener("keypress", function(event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendButton.click();
        }
    });
});