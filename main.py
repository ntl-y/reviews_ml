import requests
from bs4 import BeautifulSoup
import os
import time
import sys

# Функция для загрузки одной страницы и сохранения её в файл с использованием прокси
def download_page_with_proxy(url, filename, proxy):
    try:
        # response = requests.get(url, proxies={'http': "http://"+ proxy})
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return True
        else:
            print(response.status_code)
            print(f"Ошибка при скачивании страницы {url}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ошибка при загрузке страницы {url}: {str(e)}")
        return False

# Функция для отображения прогресса
def print_progress(current, total):
    progress = current / total * 100
    sys.stdout.write(f"\rПрогресс: {current}/{total} ({progress:.2f}%)")
    sys.stdout.flush()

# Функция для скачивания страниц из списка с использованием прокси и сменой IP при ошибке
def download_pages_from_list(file_path, output_dir, proxy_file, start_line=0):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(proxy_file, 'r') as proxy_file:
        proxy_lines = proxy_file.read().splitlines()[1:]  # Пропускаем первую строку с заголовками
        proxies = [line.split('","') for line in proxy_lines]

    urls = ["https://www.tursvodka.ru/responses/p/" + str(i) for i in range(1, 285)]

    total_pages = len(urls)
    current_proxy_index = 0
    
    for i, url in enumerate(urls, start=start_line):
        sys.stdout.write(f"\rСкачивание страницы {i}/{total_pages}")
        sys.stdout.flush()
        
        filename = os.path.join(output_dir, f'page_{i}.html')
        
        success = False
        while not success and current_proxy_index < len(proxies):
            current_proxy = proxies[current_proxy_index]
            proxy = current_proxy[0]
            
            if download_page_with_proxy(url, filename, proxy):
                success = True
            else:
                time.sleep(15)
                print("change")
                # print("CHANGED to", current_proxy_index)
                current_proxy_index += 1
        
        if success:
            pass
            # sys.stdout.write(f"\rСтраница {i} успешно скачана\n")
        else:
            sys.stdout.write(f"\rНе удалось скачать страницу {i}\n")
        
        print_progress(i, total_pages)

if __name__ == "__main__":
    input_file = "output.txt"  # Путь к файлу с ссылками
    output_directory = "html-list"  # Папка для сохранения HTML-файлов
    proxy_file = "Free_Proxy_List.txt"  # Файл с прокси-серверами
    starting_line = 0  # С какой строки начинать скачивание
    
    download_pages_from_list(input_file, output_directory, proxy_file, starting_line)
