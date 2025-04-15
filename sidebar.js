const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clear_memory_button = document.getElementById('clear-memory-button');
const clear_memory_status = document.getElementById("clear-memory-status");
const chatContainer = document.getElementById('chat-container');
let selectedModel = null;
let waitingDots = null;
const modelSelect = document.getElementById('model-select');

// 告訴 content.js 我準備好了
window.parent.postMessage({ source: 'iframe-ai-assistant', type: 'iframeReady' }, '*');

sendButton.addEventListener('click', () => {
    sendButton.disabled = true;  // 禁用按鈕
    const question = userInput.value.trim();
    if (question === '') {
        sendButton.disabled = false;
        return;
    }

    appendMessage('user', question);
    userInput.value = '';
    userInput.focus();

    // 發送訊息給 content.js
    window.parent.postMessage({
        source: 'iframe-ai-assistant',
        type: 'queryAI',
        payload: {
            question: question,
            model_name: selectedModel
        }
    }, '*');
    
    waitingDots = createWaitingDots();
    chatContainer.appendChild(waitingDots);
});

clear_memory_button.addEventListener("click", async () => {
    clear_memory_button.disabled = true;  // 禁用按鈕
    clear_memory_status.textContent = "Clearing memory...";
    clear_memory_status.style.color = "gray";

    window.parent.postMessage({
        source: 'iframe-ai-assistant',
        type: 'clearMemory',
    }, '*');
});

function appendMessage(role, content, modelName = null) {
    const message = document.createElement('div');
    message.classList.add('message', role);

    const contentElement = document.createElement('div');
    contentElement.classList.add('content');

    if (role === 'ai') {
        chatContainer.removeChild(waitingDots);
        const rawHTML = marked.parse(content);
        const safeHTML = DOMPurify.sanitize(rawHTML);
        contentElement.innerHTML = safeHTML;
    } else if (role === 'user') {
        contentElement.textContent = content;
    }

    message.appendChild(contentElement);
    chatContainer.appendChild(message);

    // 建立 footer（時間戳 + 模型名）
    const footer = document.createElement('div');
    footer.classList.add('timestamp', role);
    

    const timeString = new Date().toLocaleString('zh-TW', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    
    footer.textContent = timeString;

    if (role === 'ai' && modelName) {
        const modelTag = document.createElement('span');
        modelTag.classList.add('model-tag');
        modelTag.textContent = ` • ${modelName}`;
        modelTag.style.marginLeft = '8px';
        footer.appendChild(modelTag);
    }

    // 插入到訊息框「下方」
    chatContainer.insertBefore(footer, message.nextSibling);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function createWaitingDots() {
    const dotContainer = document.createElement('div');
    dotContainer.className = 'waiting-dots';

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        dot.className = 'dot';
        dot.style.animationDelay = `${i * 0.2}s`;
        dotContainer.appendChild(dot);
    }

    return dotContainer;
}


// 接收 content.js 的回應
window.addEventListener('message', (event) => {
    if (!event.data || event.data.source !== 'content-script-ai-assistant') return;

    const { type, payload } = event.data;

    switch(type) {
        case 'aiResponse':
            appendMessage('ai', payload.response, payload.model_name || null);
            sendButton.disabled = false; // 啟用按鈕
            break;
        case 'aiError':
            appendMessage('ai', `❗ Error: ${payload.response}`);
            sendButton.disabled = false; // 啟用按鈕
            break;
        case 'noContent':
            appendMessage('ai', `⚠️ ${payload.message}`);
            sendButton.disabled = false; // 啟用按鈕
            break;
        case 'memoryCleared':
            clear_memory_status.textContent = payload.message;
            clear_memory_status.style.color = "green";
            clear_memory_button.disabled = false; // 啟用按鈕
            break;
        case 'clearMemoryError':
            clear_memory_status.textContent = payload.response;
            clear_memory_status.style.color = "red";
            clear_memory_button.disabled = false; // 啟用按鈕
            break;
        default:
            console.warn("Unknown message type from content.js:", type);
    }
});

modelSelect.addEventListener('change', (e) => {
    selectedModel = e.target.value;
    // 可選：儲存選擇到 storage
    chrome.storage.local.set({ selectedModel });
});
