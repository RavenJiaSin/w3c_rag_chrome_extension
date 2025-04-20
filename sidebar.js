const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clear_memory_button = document.getElementById('clear-memory-button');
const chatContainer = document.getElementById('chat-container');
const modelSelect = document.getElementById('model-select');
let selectedModel = null;
let waitingDots = null;

// 於chatContainer中加入新訊息
function appendMessage(role, content, modelName = null) {
    // 創建新的訊息泡泡
    const message = document.createElement('div');
    message.classList.add('message', role);

    // 創建新的文字訊息
    const contentElement = document.createElement('div');
    contentElement.classList.add('content');

    // 判斷當前訊息是由ai或user發出
    if (role === 'ai') {
        // 移除聊天室中的等待符號
        chatContainer.removeChild(waitingDots);
        // 處理markdown
        const rawHTML = marked.parse(content);
        const safeHTML = DOMPurify.sanitize(rawHTML);
        contentElement.innerHTML = safeHTML;
    } else if (role === 'user') {
        contentElement.textContent = content;
    } else if (role === 'system') {
        contentElement.textContent = content;
    }
    

    // 將文字訊息放入訊息泡泡
    message.appendChild(contentElement);
    // 將訊息泡泡放入聊天室
    chatContainer.appendChild(message);

    // 建立 footer（時間戳 + 模型名）
    const footer = document.createElement('div');
    footer.classList.add('timestamp', role);
    
    // 時間格式
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

// 等待ai回應時的圖示
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

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.className = 'toast hidden';
    }, 3000); // 顯示 3 秒
}

function initialize() {
    // 告訴 content.js 準備好了
    window.parent.postMessage({ source: 'iframe-ai-assistant', type: 'iframeReady' }, '*');
    const welcomeMessage = `歡迎使用 W3C 標準閱讀助手！請在下方輸入您對當前頁面標準文件的任何問題，例如：「這段規範的意思是什麼？」或「請用白話解釋這個定義」。系統將根據頁面內容與相關技術文件，即時為您提供 AI 解說。`;
    appendMessage('system', welcomeMessage);
    return;
}


initialize();

// 按下訊息傳送按鈕
sendButton.addEventListener('click', () => {
    sendButton.disabled = true;  // 禁用按鈕
    const question = userInput.value.trim();
    if (question === '') { // 若訊息為空不會船到後台
        sendButton.disabled = false;
        return;
    }

    // 將user訊息放入聊天室
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
    
    // 新增等待圖示到聊天室
    waitingDots = createWaitingDots();
    chatContainer.appendChild(waitingDots);
});

// 按下記憶清除按鈕
clear_memory_button.addEventListener("click", async () => {
    clear_memory_button.disabled = true;  // 禁用按鈕
    clear_memory_button.textContent = "Clearing ...";
    

    window.parent.postMessage({
        source: 'iframe-ai-assistant',
        type: 'clearMemory',
    }, '*');
});


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
            clear_memory_button.textContent = "Clear Memory";
            clear_memory_button.disabled = false; // 啟用按鈕
            showToast("記憶已成功清除！", "success");
            break;
        case 'clearMemoryError':
            clear_memory_button.textContent = "Clear Memory";
            clear_memory_button.disabled = false; // 啟用按鈕
            showToast("清除記憶失敗，請稍後再試！", "error");
            break;
        default:
            console.warn("Unknown message type from content.js:", type);
    }
});

// 選擇模型
modelSelect.addEventListener('change', (e) => {
    selectedModel = e.target.value;
    // 可選：儲存選擇到 storage
    // chrome.storage.local.set({ selectedModel });
});
