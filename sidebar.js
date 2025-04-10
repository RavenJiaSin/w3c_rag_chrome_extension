const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const chatContainer = document.getElementById('chat-container');

// 告訴 content.js 我準備好了
window.parent.postMessage({ source: 'iframe-ai-assistant', type: 'iframeReady' }, '*');

sendButton.addEventListener('click', () => {
    const question = userInput.value.trim();
    if (question === '') return;

    appendMessage('user', question);
    userInput.value = '';
    userInput.focus();

    // 發送訊息給 content.js
    window.parent.postMessage({
        source: 'iframe-ai-assistant',
        type: 'queryAI',
        payload: { question }
    }, '*');
});

function appendMessage(role, content) {
    const message = document.createElement('div');
    message.classList.add('message', role);

    // 創建時間戳記元素，並將其顯示在對話框外
    // 獲取當前時間戳記
    const timestamp = new Date().toLocaleString(); // 使用本地時間格式（你可以根據需要調整格式）
    const timestampElement = document.createElement('div');
    timestampElement.classList.add('timestamp', role);
    timestampElement.textContent = timestamp;

    if (role === 'ai') {
        // 將 Markdown 轉換為 HTML，並清理 XSS 風險
        const rawHTML = marked.parse(content);
        const safeHTML = DOMPurify.sanitize(rawHTML);
        message.innerHTML = safeHTML;
        message.classList.add('message', 'ai');
        timestampElement.classList.add('timestamp','ai');
    } else {
        message.textContent = content;
        message.classList.add('message', 'user');
        timestampElement.classList.add('timestamp','user');
    }

    chatContainer.appendChild(message);

    // 將時間戳記插入到對話框外，並放置於訊息上方
    message.parentNode.appendChild(timestampElement);
    

    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 接收 content.js 的回應
window.addEventListener('message', (event) => {
    if (!event.data || event.data.source !== 'content-script-ai-assistant') return;

    const { type, payload } = event.data;

    switch(type) {
        case 'aiResponse':
            appendMessage('ai', payload.response);
            break;
        case 'aiError':
            appendMessage('ai', `❗ Error: ${payload.response}`);
            break;
        case 'noContent':
            appendMessage('ai', `⚠️ ${payload.message}`);
            break;
        default:
            console.warn("Unknown message type from content.js:", type);
    }
});
