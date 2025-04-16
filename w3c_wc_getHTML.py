import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import sqlite3
from tqdm import tqdm

# 設定 WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 無頭模式

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 目標網址
url = "https://www.w3.org/TR/"
driver.get(url)
driver.implicitly_wait(10)

# 建立 SQLite 資料庫
conn = sqlite3.connect("w3c_data.db")
cursor = conn.cursor()

# 建立DB資料表
cursor.execute('''
CREATE TABLE IF NOT EXISTS w3c_standards (
    Title TEXT PRIMARY KEY,
    Content TEXT
)
''')
conn.commit()

# **尋找所有standers element**
try:
    standers_element_list = driver.find_elements(By.XPATH, "/html/body/div/div/main/div[3]/section/div/div/h3/a")
    # **由舊到新排列**
    standers_element_list.reverse()
    standers_href_list = [element.get_attribute("href") for element in standers_element_list]
    print(f"===找到 {len(standers_href_list)} 行資料===")
    print("最舊一筆:",standers_href_list[0])
    print("最新一筆:",standers_href_list[-1])
except Exception as e:
    print(f"ERROR:{e}")
    driver.quit()
    exit()

# **獲取standers**
def get_stander(stander_url):
    driver.get(stander_url)
    driver.implicitly_wait(10)
    title = driver.title
    print(f"<{title}>")
    print("got title...")
    content = " ".join([elem.text for elem in driver.find_elements(By.XPATH, "//p | //div")])
    #print(content)
    print("got contents...")
    
    # **存入DB**
    cursor.execute('''
    INSERT INTO w3c_standards (
        Title, 
        Content
    ) VALUES (?, ?)
    ON CONFLICT(Title) DO UPDATE SET 
        Content = excluded.Content
    ''', (title,content))
    print(f"---Successfully store in DB :{title}---")

    # 下載HTML
    try:
        response = requests.get(stander_url, timeout=10)
        response.raise_for_status()  # 確保請求成功
        print(stander_url)
        with open(f'w3c_html/{stander_url[22:-1]}.html', "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"網頁 HTML 已成功下載並儲存為 {stander_url[22:-1]+'.html'}")
    except requests.exceptions.RequestException as e:
        print(f"下載失敗: {e}")

    
    

# **進度選擇**
start_point = int(input(f"請輸入更新起始點(舊0-->{len(standers_href_list)-1}新, 從頭爬取請輸入0):"))
#start_point = 1162 #測試用

# **爬取standers**
try:
    for nth_stander in tqdm(range(start_point, len(standers_href_list))):
        nth_stander_href = standers_href_list[nth_stander]
        try:
            print(f"---Crawling <{nth_stander}> stander---")
            get_stander(nth_stander_href)
        except Exception as e:
            print(f"ERROR:{e}")
            print(f"本次爬取資料區間:{start_point}-{nth_stander-1} / 0-{len(standers_href_list)-1}")
            exit()
    print("===已成功將所有資料存入DB===")
    print(f"本次爬取資料區間:{start_point}-{len(standers_href_list)-1} / 0-{len(standers_href_list)-1}")
except Exception as e:
    print(f"ERROR:{e}")


# **提交並關閉資料庫**
conn.commit()
conn.close()

# **關閉瀏覽器**
driver.quit()

