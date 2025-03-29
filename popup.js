document.addEventListener("DOMContentLoaded", function () {
    const inputField = document.getElementById("question-input");
    const sendButton = document.getElementById("send-button");
    const responseDiv = document.getElementById("response");

    sendButton.addEventListener("click", function () {
        const userQuestion = inputField.value.trim();
        if (!userQuestion) return;

        responseDiv.innerText = "📡 正在查詢 AI...";
        
        chrome.runtime.sendMessage({ action: "queryOllama", question: userQuestion }, (response) => {
            if (chrome.runtime.lastError) {
                responseDiv.innerText = "❌ 擴充功能錯誤：" + chrome.runtime.lastError.message;
            } else {
                responseDiv.innerText = "💬 AI 回覆：" + response.response;
            }
        });
    });
});
