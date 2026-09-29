# download_images.py
import os
import requests
from bs4 import BeautifulSoup
import time

def download_google_images(search_term, save_dir, num_images=50):
    """Google'dan resim indir"""
    os.makedirs(save_dir, exist_ok=True)
    
    url = f"https://www.google.com/search?q={search_term}&tbm=isch"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Resim linklerini bul
    img_tags = soup.find_all('img')
    count = 0
    
    for img in img_tags:
        if count >= num_images:
            break
        try:
            img_url = img.get('src')
            if img_url and img_url.startswith('http'):
                img_data = requests.get(img_url).content
                with open(f"{save_dir}/{search_term}_{count}.jpg", 'wb') as f:
                    f.write(img_data)
                count += 1
                print(f"İndirildi: {search_term}_{count}.jpg")
                time.sleep(0.5)
        except:
            continue

# Kullanım
animals = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
for animal in animals:
    download_google_images(animal, f'images/{animal}', 50)