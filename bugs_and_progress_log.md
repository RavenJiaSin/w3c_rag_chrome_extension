---
title: wrce_bugs_and_progress_log

---

# w3c rag chrome extension 踩坑日誌

---
3/28

:::danger
遇到問題:
ollama一直403，查了[資料](https://stackoverflow.com/questions/77911717/issue-with-calling-a-local-ollama-api-from-chrome-extension)後發現是要設定`OLLAMA_ORIGIN` 來允許來自 chrome 擴充功能來源的請求，文章中說的:
```
OLLAMA_ORIGINS=chrome-extension://* ollama serve
```
chatGPT說在PowerShell中要打成:
```
$env:OLLAMA_ORIGINS="chrome-extension://*"
ollama serve
```
發現在serve完後開chrome extension可以正常運行，但關掉powershell就壞了，因此要永久設置環境變數，用系統管理員開啟powershell輸入:
```
[System.Environment]::SetEnvironmentVariable("OLLAMA_ORIGINS", "chrome-extension://*", "User")
```
重新啟動後執行`$env:OLLAMA_ORIGINS`，若有`chrome-extension://*`，代表設置成功





:::

:::info
目前進度:
能連接llm，還沒有解析w3c網頁以及RAG功能
![image](https://hackmd.io/_uploads/BycR-HVTyg.png)
:::

---
3/29-1
:::info
目前進度:能擷取網頁文字
![image](https://hackmd.io/_uploads/BkAFmkS61x.png)
:::
3/26-2
:::info
目前進度，能根據抓到的網頁(standar)與使用者提問回答一個問題(還沒有記憶，還沒使用RAG)
等待一次回應時間不超過1分鐘
![image](https://hackmd.io/_uploads/r1FbyXSake.png)

:::