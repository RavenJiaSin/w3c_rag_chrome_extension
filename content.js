(() => {
    let iframe = null; // 引用注入的 iframe
    let iframeReady = false; // 標記 iframe 是否已加載並準備好接收消息
    const IFRAME_SOURCE = 'sidebar.html';

    // --- 創建並注入 iframe ---
    function injectAssistantIframe() {
        if (document.getElementById('w3c-ai-assistant-iframe')) {
            return; // 防止重複注入
        }

        iframe = document.createElement('iframe');
        iframe.id = 'w3c-ai-assistant-iframe';
        iframe.src = chrome.runtime.getURL(IFRAME_SOURCE);

        // --- 設置 iframe 樣式 ---
        iframe.style.position = 'fixed';
        iframe.style.top = '80px';
        iframe.style.right = '20px';
        iframe.style.bottom = '40px'; // 控制底部距離
        iframe.style.width = '380px'; // 與 sidebar.html 內 body 寬度匹配或稍大
        iframe.style.height = 'calc(100vh - 120px)'; // 高度基於 top 和 bottom
        iframe.style.border = 'none'; // 通常不需要邊框
        iframe.style.borderRadius = '8px'; // 可以給 iframe 加圓角
        iframe.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        iframe.style.zIndex = '9999';
        iframe.style.backgroundColor = 'white'; // 背景色以防加載時透明

        document.body.appendChild(iframe);
        console.log("✅ AI Assistant iframe injected.");
    }

    // --- 文本提取 (保持不變) ---
    function extractTextAndSend() {
       let w3cContent = "";
       let contentArray = [];
        let mainContent = document.querySelector('#main') || document.querySelector('#content') || document.body;
        const elements = mainContent.querySelectorAll('p, div:not(div div)');
        elements.forEach(node => {
             // ** 確保排除 iframe **
             if (node.closest('nav, header, footer, .toc, #toc, #w3c-ai-assistant-iframe')) {
                 return;
             }
             let text = node.innerText?.trim();
             if (text) {
                 contentArray.push(text);
             }
         });
        w3cContent = contentArray.join("\n\n");

        if (w3cContent.length > 0) {
             // console.log("✅ 成功擷取內容，長度:", w3cContent.length);
             chrome.runtime.sendMessage({ action: "extractText", content: w3cContent }, response => {
                 if (chrome.runtime.lastError) {
                     console.error("發送內容到 background 失敗:", chrome.runtime.lastError.message);
                 } else {
                     // console.log("💬 內容已發送至 background:", response?.status);
                 }
             });
         } else {
             console.warn("❌ 未能提取到頁面內容");
         }
    }

    // --- 向 iframe 發送消息 ---
    function sendMessageToIframe(type, payload) {
        if (iframe && iframe.contentWindow && iframeReady) {
             // 使用 iframe 的 contentWindow 發送消息
             // '*' 為了簡單，生產環境應指定 iframe 的源 (chrome.runtime.getURL('/') 開頭)
            iframe.contentWindow.postMessage({ source: 'content-script-ai-assistant', type: type, payload: payload }, '*');
        } else {
            console.warn("Attempted to send message to iframe, but it's not ready or not found.");
        }
    }


    // --- 監聽來自 iframe 的消息 ---
    window.addEventListener('message', (event) => {
        // 安全檢查：確保消息來自我們創建的 iframe
        if (event.source !== iframe?.contentWindow || !event.data || event.data.source !== 'iframe-ai-assistant') {
            return;
        }

        const message = event.data;
        console.log("Content Script: Received message from iframe:", message);

        switch(message.type) {
            case 'queryAI':
                // iframe 請求查詢 AI
                const question = message.payload.question;
                console.log("Content Script: Forwarding query to background:", question);
                // **向 background 發送查詢請求**
                chrome.runtime.sendMessage({ action: "queryGemini", question: question }, (response) => {
                     if (chrome.runtime.lastError) {
                         console.error("Content Script: Error communicating with background:", chrome.runtime.lastError.message);
                         // **將錯誤信息發回給 iframe**
                         sendMessageToIframe('aiError', { status: 'error', response: `通訊錯誤: ${chrome.runtime.lastError.message}` });
                         return;
                     }

                     if (response) {
                        // **將 background 的回應轉發給 iframe**
                        console.log("Content Script: Received response from background, forwarding to iframe:", response);
                        // 發送完整的響應對象，讓 iframe 根據 status 處理
                         if (response.status === 'success') {
                            sendMessageToIframe('aiResponse', { status: response.status, response: response.response });
                         } else if (response.status === 'no_content') {
                             sendMessageToIframe('noContent', { status: response.status, message: response.message });
                         } else if (response.status === 'error') {
                             sendMessageToIframe('aiError', { status: response.status, response: response.response });
                         } else {
                             // 其他未知狀態
                             sendMessageToIframe('unknownStatus', { status: response.status, payload: response });
                         }

                     } else {
                         console.error("Content Script: Received invalid response from background.");
                          sendMessageToIframe('aiError', { status: 'error', response: '收到來自背景的無效回應' });
                     }
                 });
                break;
            case 'iframeReady':
                // iframe 發送消息表示它已加載完成
                console.log("Content Script: iframe reported ready.");
                iframeReady = true;
                // 可以在這裡做一些 iframe 加載後才做的事情
                break;
            // 可以添加其他消息類型
        }
    });


    // --- 初始化 ---
    function initialize() {
        injectAssistantIframe(); // 注入 iframe
        extractTextAndSend();    // 提取文本
    }

    // --- 啟動 ---
    if (document.readyState === "complete" || document.readyState === "interactive") {
        initialize();
    } else {
        document.addEventListener('DOMContentLoaded', initialize);
    }

})();