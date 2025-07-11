import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

df = pd.read_csv("D:\proga\\test3\data\URL_list.csv")
products = []

for url in df["url"]:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        title = soup.find("h1").get_text().strip()
        products.append({"text": title, "entities": []})

    except Exception as e:
        print(f"Ошибка при парсинге {url}: {e}")

with open("D:\proga\\test3\data\data.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False)