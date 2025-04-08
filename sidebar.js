const botui = new BotUI('botui-app');

// 啟動對話
botui.message.add({
  content: '您好，我是您的 AI 助手！請問有什麼可以幫您？'
}).then(askQuestion);

function askQuestion() {
  botui.action.text({
    action: {
      placeholder: '請輸入您的問題...'
    }
  }).then(res => {
    const question = res.value;
    
    // 顯示使用者輸入
    botui.message.add({ human: true, content: question });

    // 通知 content.js
    window.parent.postMessage({
      source: 'iframe-ai-assistant',
      type: 'queryAI',
      payload: { question }
    }, '*');

    // 等待回答中訊息
    botui.message.add({
      content: '思考中...'
    }).then(() => {});
  });
}

// 接收 content.js 的回應
window.addEventListener("message", (event) => {
  if (event.data?.source !== 'content-script-ai-assistant') return;

  const { type, payload } = event.data;

  if (type === 'aiResponse') {
    // 移除“思考中”訊息並顯示回答
    botui.message.removeAll().then(() => {
      botui.message.add({ content: payload.response }).then(askQuestion);
    });
  } else if (type === 'aiError') {
    botui.message.add({ content: `錯誤：${payload.response}` }).then(askQuestion);
  } else if (type === 'noContent') {
    botui.message.add({ content: '⚠️ 抱歉，無法提取頁面內容。' }).then(askQuestion);
  }
});

// 告知 content.js iframe 已準備好
window.parent.postMessage({
  source: 'iframe-ai-assistant',
  type: 'iframeReady'
}, '*');
