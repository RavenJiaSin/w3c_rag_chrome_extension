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
        document.body.appendChild(iframe);
        console.log("✅ AI Assistant iframe injected.");
    }
    
    // --- 注入控制按鈕 ---
    function injectToggleButton() {
        if (document.getElementById('w3c-ai-toggle-button')) {
            return; // 防止重複注入
        }

        const button = document.createElement('button');
        button.id = 'w3c-ai-toggle-button';
        button.textContent = '🤖'; 
        button.title = 'Toggle Assistant';
        // 點擊切換 iframe 顯示
        button.addEventListener('click', () => {
            const iframe = document.getElementById('w3c-ai-assistant-iframe');
            if (!iframe) return;
            const visible = iframe.style.display !== 'none';
            iframe.style.display = visible ? 'none' : 'block';
            button.textContent = visible ? '🤖' : '🤖'; // 可自行調整圖示
        });
        document.body.appendChild(button);
        console.log("✅ Toggle button injected.");
    }

    // --- 文本提取 (保持不變) ---
    function extractTextAndSend() {
        let w3cContent = "";
        let contentArray = [];
    
        // 擷取主內容區塊
        let mainContent = document.querySelector('#main') || document.querySelector('#content') || document.body;
        const elements = mainContent.querySelectorAll('p, div:not(div div)');
        elements.forEach(node => {
            if (node.closest('nav, header, footer, .toc, #toc, #w3c-ai-assistant-iframe')) {
                return;
            }
            let text = node.innerText?.trim();
            if (text) {
                contentArray.push(text);
            }
        });
        w3cContent = contentArray.join("\n\n");
    
        // 擷取文章標題：先看是否有 <h1>，否則退而求其次用 <title>
        let articleTitle = document.querySelector('h1')?.innerText?.trim()
            || document.title?.trim()
            || "Untitled";
    
        // 儲存到 Chrome Storage
        chrome.storage.local.set({ w3cContent, articleTitle }, () => {
            if (chrome.runtime.lastError) {
                console.error("❌ 儲存 W3C 內容到 storage 失敗:", chrome.runtime.lastError.message);
            } else {
                console.log("✅ 已儲存 W3C 內容與標題到 chrome.storage.local，長度:", w3cContent.length);
                console.log("📌 標題:", articleTitle);
            }
        });
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

        switch (message.type) {
            case 'queryAI':
                // iframe 請求查詢 AI
                const question = message.payload.question;
                const modelName = message.payload.model_name || null; // 可支援模型選擇

                console.log("Content Script: Forwarding query to background:", question);

                // 組裝發送資料
                const sendData = {
                    action: "queryMessage",
                    user_question: question
                };
                if (modelName) {
                    sendData.model_name = modelName;
                }

                // 向 background 發送查詢請求
                chrome.runtime.sendMessage(sendData, (response) => {
                    if (chrome.runtime.lastError) {
                        console.error("Content Script: Error communicating with background:", chrome.runtime.lastError.message);
                        sendMessageToIframe('aiError', { status: 'error', response: `通訊錯誤: ${chrome.runtime.lastError.message}` });
                        return;
                    }

                    if (response) {
                        console.log("Content Script: Received response from background, forwarding to iframe:", response);

                        if (response.status === 'success') {
                            sendMessageToIframe('aiResponse', { status: response.status, response: response.response, model_name: response.model_name || null });
                        } else if (response.status === 'no_content') {
                            sendMessageToIframe('noContent', { status: response.status, message: response.message });
                        } else if (response.status === 'error') {
                            sendMessageToIframe('aiError', { status: response.status, response: response.response });
                        } else {
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
        injectToggleButton();    // 注入 toggle 按鈕
        extractTextAndSend();    // 提取文本
    }

    // --- 啟動 ---
    if (document.readyState === "complete" || document.readyState === "interactive") {
        initialize();
    } else {
        document.addEventListener('DOMContentLoaded', initialize);
    }

})();