import os
import csv
from bs4 import BeautifulSoup

# Создайте CSV-файл для записи данных
csv_file = open('reviews.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file, delimiter=';')
csv_writer.writerow(['Идентификатор', 'Оценка', 'Комментарий'])

# Перебирайте файлы в папке html-list
html_folder = 'html-list'
for filename in os.listdir(html_folder):
    print(filename)
    if filename.endswith('.html'):
        html_path = os.path.join(html_folder, filename)

        # Откройте HTML-файл и считайте его содержимое
        with open(html_path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()

        # Используйте BeautifulSoup для парсинга HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Найдите все отзывы на странице
        reviews = soup.find_all('div', class_='response__descript')

        # Перебирайте отзывы
        for i, review in enumerate(reviews, start=1):
            # Найдите оценку для отзыва
            rating_div = review.find_previous('div', class_='listrating__rate')
            if rating_div:
                rating = float(rating_div.text.strip())
            else:
                rating = None

            # Извлеките и объедините комментарий для отзыва (включая абзацы)
            comment = '\n'.join(p.text.strip() for p in review.find_all('p'))

            # Замените переводы строк на пробелы в комментарии
            comment = comment.replace('\n', ' ')

            # Запишите данные в CSV-файл
            csv_writer.writerow([f'{filename}_{i}', rating, comment])

# Закройте CSV-файл
csv_file.close()

print('Данные успешно записаны в reviews.csv')
