(() => {
    function extractText() {
        let contentArray = [];

        // 使用 XPath 抓取所有 <p> 和 <div> 元素
        let xpathQuery = "//p | //div";
        let elements = document.evaluate(xpathQuery, document, null, XPathResult.ORDERED_NODE_ITERATOR_TYPE, null);

        let node;
        while ((node = elements.iterateNext())) {
            let text = node.innerText.trim();
            if (text) {
                contentArray.push(text);
            }
        }

        let textContent = contentArray.join("\n");

        if (textContent.length === 0) {
            console.warn("❌ 無法找到主要內容");
        } else {
            console.log("✅ 成功擷取標準內容，長度:", textContent.length);
        }

        // 發送到 background.js
        chrome.runtime.sendMessage({ action: "extractText", content: textContent }, (response) => {
            console.log("✅ 文字內容已傳送到 background.js:", response);
        });
    }

    // 執行擷取
    extractText();
})();
