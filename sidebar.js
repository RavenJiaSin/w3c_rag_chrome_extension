// 獲取 iframe 內的元素
const inputField = document.getElementById("sidebar-question-input");
const sendButton = document.getElementById("sidebar-send-button");
const responseDiv = document.getElementById("sidebar-response");

// --- 向 Content Script (父頁面) 發送消息 ---
function sendMessageToContentScript(type, payload) {
    // 向父窗口發送消息，指定目標源確保安全 (或者使用 '*')
    // '*' 為了簡單，實際應用可能需要更精確的 origin
    window.parent.postMessage({ source: 'iframe-ai-assistant', type: type, payload: payload }, '*');
}

// --- 處理按鈕點擊 ---
sendButton.addEventListener("click", () => {
    const userQuestion = inputField.value.trim();
    if (!userQuestion) {
        responseDiv.innerHTML = "<span style='color: orange;'>⚠️ 請先輸入問題。</span>";
        return;
    }

    // 顯示等待狀態
    responseDiv.innerHTML = "<span style='color: gray;'>⏳ 正在準備請求...</span>";
    sendButton.disabled = true;
    inputField.disabled = true;

    // 通過 postMessage 將問題發送給 content script
    sendMessageToContentScript('queryAI', { question: userQuestion });
});

// --- 監聽來自 Content Script 的消息 ---
window.addEventListener('message', (event) => {
    // 基本的安全檢查：確保消息來自預期的來源 (父窗口) 並且是我們的擴充功能發出的
    // 注意：event.source !== window.parent 的檢查可能不總是在所有瀏覽器或場景下可靠
    // 更好的做法是檢查 event.origin (如果 content script 在發送時指定了 targetOrigin)
    // 或者檢查消息內容中的特定標識符
    if (event.source !== window.parent || !event.data || event.data.source !== 'content-script-ai-assistant') {
       // console.log("Sidebar: Ignoring message from unknown source:", event.origin, event.data);
        return;
    }

    const message = event.data;
    console.log("Sidebar: Received message from content script:", message);

    // 根據消息類型處理
    switch (message.type) {
        case 'aiResponse':
            // 收到 AI 的原始回應 (Markdown 文本)
            handleAIResponse(message.payload.status, message.payload.response);
            break;
        case 'aiError':
            // 收到來自 Background 的錯誤
            handleAIResponse(message.payload.status, message.payload.response);
            break;
        case 'noContent':
            // 收到 "尚未提取內容" 的提示
             handleAIResponse(message.payload.status, message.payload.message);
             break;
        // 可以添加其他消息類型
    }
});

// --- 處理並顯示 AI 回應 ---
function handleAIResponse(status, data) {
    // 啟用按鈕和輸入框
    sendButton.disabled = false;
    inputField.disabled = false;

    switch (status) {
        case "success":
            console.log("Sidebar: Rendering AI response...");
            // **在這裡使用 marked 和 DOMPurify**
            try {
                // **直接調用，無需 window. 前綴**
                const rawHtml = marked.parse(data || "");
                const cleanHtml = DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true }, ADD_ATTR: ['target'] });
                responseDiv.innerHTML = cleanHtml;
                // **不再需要 applyMarkdownStyles，樣式在 HTML/CSS 中**
            } catch (renderError) {
                console.error("Sidebar: Markdown rendering failed:", renderError);
                responseDiv.innerHTML = `<span style='color: red;'>❌ 無法渲染回應。<br>錯誤：${renderError.message}</span>`;
            }
            break;
        case "no_content":
            responseDiv.innerHTML = `<span style='color: orange;'>🟡 ${data}</span>`;
            break;
        case "error":
            responseDiv.innerHTML = `<span style='color: red;'>❌ 錯誤：${data}</span>`;
            break;
        default:
            responseDiv.innerHTML = `<span style='color: orange;'>❓ 未知狀態：${status}</span>`;
    }
     // 自動滾動到底部 (可選)
     responseDiv.scrollTop = responseDiv.scrollHeight;
}

// --- 輔助函數：轉義 HTML (如果需要在錯誤時顯示原始文本) ---
// function escapeHTML(str) { ... } // 可以保留以備不時之需

console.log("Sidebar script loaded.");
// 可以發送一個消息告訴 content script iframe 已加載完成 (可選)
sendMessageToContentScript('iframeReady', {});