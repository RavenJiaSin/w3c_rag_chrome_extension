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

    if (role === 'ai') {
        // 將 Markdown 轉換為 HTML，並清理 XSS 風險
        const rawHTML = marked.parse(content);
        const safeHTML = DOMPurify.sanitize(rawHTML);
        message.innerHTML = safeHTML;
        message.classList.add('message', 'ai');
        message.style.alignSelf = 'flex-start';
    } else {
        message.textContent = content;
        message.classList.add('message', 'user');
        message.style.alignSelf = 'flex-end';
    }

    chatContainer.appendChild(message);
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
