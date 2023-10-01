Импортирование всех библиотек


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
```


```python
# чтение данных из файла reviews.csv с использованием точки с запятой в качестве разделителя
data = pd.read_csv("reviews.csv", delimiter=";")

# вывод информации о структуре
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8343 entries, 0 to 8342
    Data columns (total 3 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Идентификатор  8343 non-null   object 
     1   Оценка         8248 non-null   float64
     2   Комментарий    8343 non-null   object 
    dtypes: float64(1), object(2)
    memory usage: 195.7+ KB
    

Нормальзация данных


```python
# удаление столбца Идентификатор
data.drop(columns=['Идентификатор'], inplace=True)

# вывод первых нескольких строк
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Оценка</th>
      <th>Комментарий</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>Мошенники! Не связывайтесь с ними! Вечером нак...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>Если оплатили деньги за путёвку, но изменились...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.7</td>
      <td>Этот отель мы выбрали, поскольку был опыт отли...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>Это был мой лучший отдых, благодаря этой компа...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.0</td>
      <td>Отличная компания для организации отдыха в ОАЭ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# удаление строк с отсутствующими значениями в столбце Оценка
data = data.dropna(subset=["Оценка"])

# удаление строк с отсутствующими значениями в столбце Комментарий
data = data.dropna(subset=["Комментарий"])

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8248 entries, 0 to 8342
    Data columns (total 2 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Оценка       8248 non-null   float64
     1   Комментарий  8248 non-null   object 
    dtypes: float64(1), object(1)
    memory usage: 193.3+ KB
    


```python
sns.countplot(x='Оценка', data=data, color='skyblue');
```


    
![png](README_files/README_6_0.png)
    


Оценки 10.0 очень много уберем половину


```python
data10 = data[data['Оценка'] == 10].sample(frac=0.5, random_state=42)
data = data.drop(data10.index)
```


```python
# преобразуем столбец 'Оценка' к целочисленному типу данных
data['Оценка'] = data['Оценка'].astype(int)

# группируем данные по оценкам и подсчитываем количество комментариев для каждой оценки
comments_by_rating = data['Оценка'].value_counts().sort_index()

# создаем график
plt.figure(figsize=(8, 6))
plt.bar(comments_by_rating.index, comments_by_rating.values, color='skyblue')
plt.xlabel('Оценка')
plt.ylabel('Количество комментариев')
plt.title('Количество комментариев по каждой оценке')
plt.xticks(comments_by_rating.index)
plt.show()
```


    
![png](README_files/README_9_0.png)
    


У меня есть оценки  3.1, 4.3, 0.1
Это очень много. Округлю оценнки и удалю нулевые оценки для удобства.


```python
# округляем столбец 'Оценка' до целых чисел
data['Оценка'] = data['Оценка'].round().astype(int)
# удаляем строки, в которых оценка равна нулю
data = data.query('Оценка != 0')
```


```python
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Оценка', palette='viridis')
plt.xlabel('Оценка')
plt.ylabel('Количество комментариев')
plt.title('Распределение комментариев по округленным оценкам (без нулей)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```


    
![png](README_files/README_12_0.png)
    



```python
# подсчитываем количество оценок каждого значения
count_by_rating = data['Оценка'].value_counts().sort_index()
```


```python
# Создаем график
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Оценка', palette='viridis', order=data['Оценка'].value_counts().index)
plt.xlabel('Оценка')
plt.ylabel('Количество оценок')
plt.title('Количество оценок по каждой оценке')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```


    
![png](README_files/README_14_0.png)
    



```python
# сбрасываем индекс DataFrame, чтобы начать с 0 и иметь более структурированный индекс
data = data.reset_index()

# удаляем столбец "index", который может быть лишним после сброса индекса
data = data.drop(columns="index")
```

Для создания облака часто встречающихся слов (или Word Cloud) из комментариев, а также для деления на негативные и позитивные комментарии, мы сначала должны провести лемматизацию и анализ тональности комментариев.


```python
# объединение все комментарии в одну строку
all_comments = " ".join(data['Комментарий'])
```


```python
# инициализация лемматизатора и стоп-слов
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(stopwords.words("russian"))
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\ASUS\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\ASUS\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\ASUS\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
# фнкция для лемматизации текста и удаления стоп-слов
def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word.lower() not in stop_words]
    return " ".join(words)
```


```python
all_comments = preprocess_text(all_comments)

# Word Cloud для всех комментариев
wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
```


```python
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud для всех комментариев')
plt.show()
```


    
![png](README_files/README_21_0.png)
    


Пусть негативные отзывы это оценки 1 2 3 4


Нейтральные отзывы 5 6 7


И положительные отзывы 8 9 10


```python
data['Лемматизированный_комментарий'] = data['Комментарий'].apply(preprocess_text)
```


```python
# функция для классификации отзывов на негативные, нейтральные и положительные
def classify_review(rating):
    if rating in [1, 2, 3, 4]:
        return 'Негативный'
    elif rating in [5, 6, 7]:
        return 'Нейтральный'
    else:
        return 'Положительный'
```


```python
data['Тип_отзыва'] = data['Оценка'].apply(classify_review)
```


```python
# Word Cloud для негативных отзывов
negative_reviews = " ".join(data[data['Тип_отзыва'] == 'Негативный']['Лемматизированный_комментарий'])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)

# Word Cloud для нейтральных отзывов
neutral_reviews = " ".join(data[data['Тип_отзыва'] == 'Нейтральный']['Лемматизированный_комментарий'])
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_reviews)

# Word Cloud для положительных отзывов
positive_reviews = " ".join(data[data['Тип_отзыва'] == 'Положительный']['Лемматизированный_комментарий'])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
```


```python
# для негативных отзывов
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud для негативных отзывов')
plt.show()

# для нейтральных отзывов
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud для нейтральных отзывов')
plt.show()

# для положительных отзывов
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud для положительных отзывов')
plt.show()
```


    
![png](README_files/README_27_0.png)
    



    
![png](README_files/README_27_1.png)
    



    
![png](README_files/README_27_2.png)
    



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Оценка</th>
      <th>Комментарий</th>
      <th>Лемматизированный_комментарий</th>
      <th>Тип_отзыва</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Мошенники! Не связывайтесь с ними! Вечером нак...</td>
      <td>Мошенники связывайтесь ними Вечером накануне о...</td>
      <td>Негативный</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Если оплатили деньги за путёвку, но изменились...</td>
      <td>оплатили деньги путёвку изменились планы либо ...</td>
      <td>Негативный</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Этот отель мы выбрали, поскольку был опыт отли...</td>
      <td>отель выбрали поскольку опыт отличного отдыха ...</td>
      <td>Негативный</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>Отличная компания для организации отдыха в ОАЭ...</td>
      <td>Отличная компания организации отдыха ОАЭ Точно...</td>
      <td>Положительный</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>Хочу поделиться о "Центре оздоровления и реаби...</td>
      <td>Хочу поделиться Центре оздоровления реабилитац...</td>
      <td>Положительный</td>
    </tr>
  </tbody>
</table>
</div>



**Использование лемматизированного текста для обучения:**

*Преимущества:*
1. **Уменьшение размерности:** Лемматизация может снизить размерность данных, поскольку слова будут представлены в более общей форме, что может улучшить производительность модели и сократить требуемое количество данных для обучения.
2. **Снижение шума:** Лемматизация может устранить различные формы слова, что помогает снизить шум в данных и сделать их более консистентными.
3. **Улучшение обобщающей способности:** За счет сокращения форм слова, модель может лучше обобщать изученные закономерности на новых данных.

*Недостатки:*
1. **Потеря информации:** В процессе лемматизации теряется некоторая информация, такая как грамматическая форма слова, которая может быть важной для определенных задач.
2. **Слова-независимые**: Лемматизация может сделать некоторые слова абсолютно независимыми друг от друга, что может быть не всегда желательно.

**Использование оригинального комментария для обучения:**

*Преимущества:*
1. **Сохранение контекста:** Оригинальные комментарии сохраняют контекст и специфику форм слов, что может быть полезным для задач, требующих сохранения этой информации.
2. **Потенциально более точные результаты:** В некоторых случаях, особенно если ваша задача требует точной обработки слов, использование оригинальных комментариев может быть предпочтительным.

*Недостатки:*
1. **Увеличение размерности:** Оригинальные комментарии могут быть более многословными и иметь большую размерность, что может потребовать больше вычислительных ресурсов для обучения моделей.
2. **Шум и нерегулярности:** Оригинальные данные могут содержать опечатки, грамматические ошибки и другие нерегулярности, которые могут усложнить анализ.

В моем случае, для обучения моделей на отзывах, лемматизированный текст, скорее всего, является предпочтительным вариантом.


```python
# векторизатор текста
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # максимальное количество признаков (слов)
```


```python
# матрица TF-IDF для текста комментариев
X = tfidf_vectorizer.fit_transform(data['Лемматизированный_комментарий'])
y = data['Оценка']
```


```python
# обучающий и тестовый наборы в соотношении 8 к 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# модель Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
```

    C:\Users\ASUS\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
# модель Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">MultinomialNB</label><div class="sk-toggleable__content"><pre>MultinomialNB()</pre></div></div></div></div></div>




```python
# модель Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>




```python
# оценка производительности моделей
y_pred_lr = lr_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("Оценка производительности модели Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}\n")

print("Оценка производительности модели Naive Bayes:")
print(classification_report(y_test, y_pred_nb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb)}\n")

print("Оценка производительности модели Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}\n")
```

    Оценка производительности модели Logistic Regression:
                  precision    recall  f1-score   support
    
               1       0.49      0.57      0.52       125
               2       0.26      0.16      0.20        89
               3       0.33      0.02      0.03        55
               4       0.11      0.02      0.04        48
               5       0.33      0.02      0.03        59
               6       0.00      0.00      0.00        65
               7       0.27      0.05      0.09        74
               8       0.24      0.18      0.21       197
               9       0.29      0.51      0.37       293
              10       0.46      0.61      0.52       323
    
        accuracy                           0.36      1328
       macro avg       0.28      0.21      0.20      1328
    weighted avg       0.32      0.36      0.31      1328
    
    Accuracy: 0.3569277108433735
    
    Оценка производительности модели Naive Bayes:
                  precision    recall  f1-score   support
    
               1       0.55      0.55      0.55       125
               2       0.28      0.08      0.12        89
               3       0.00      0.00      0.00        55
               4       0.00      0.00      0.00        48
               5       0.00      0.00      0.00        59
               6       0.00      0.00      0.00        65
               7       0.00      0.00      0.00        74
               8       0.18      0.07      0.10       197
               9       0.29      0.66      0.40       293
              10       0.47      0.62      0.53       323
    
        accuracy                           0.36      1328
       macro avg       0.18      0.20      0.17      1328
    weighted avg       0.27      0.36      0.29      1328
    
    Accuracy: 0.36370481927710846
    
    Оценка производительности модели Random Forest:
                  precision    recall  f1-score   support
    
               1       0.45      0.49      0.47       125
               2       0.16      0.10      0.12        89
               3       0.50      0.02      0.04        55
               4       0.00      0.00      0.00        48
               5       1.00      0.02      0.03        59
               6       0.12      0.02      0.03        65
               7       0.25      0.01      0.03        74
               8       0.20      0.09      0.13       197
               9       0.28      0.63      0.39       293
              10       0.50      0.57      0.53       323
    
        accuracy                           0.35      1328
       macro avg       0.35      0.20      0.18      1328
    weighted avg       0.35      0.35      0.29      1328
    
    Accuracy: 0.348644578313253
    
    

    C:\Users\ASUS\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\ASUS\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\ASUS\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
# матрицы ошибок 
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=np.arange(1, 11))
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=np.arange(1, 11))
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=np.arange(1, 11))
```


```python
# Создайте функцию для визуализации матрицы ошибок
def plot_confusion_matrix_colorful(cm, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', cbar=False, square=True, xticklabels=np.arange(1, 11), yticklabels=np.arange(1, 11))
    plt.xlabel('Предсказанные оценки')
    plt.ylabel('Истинные оценки')
    plt.title(f'Матрица ошибок для модели {model_name}')
    plt.show()
```


```python
# Визуализируйте матрицу ошибок для модели Logistic Regression
plot_confusion_matrix_colorful(cm_lr, 'Logistic Regression')

# Визуализируйте матрицу ошибок для модели Naive Bayes
plot_confusion_matrix_colorful(cm_nb, 'Naive Bayes')

# Визуализируйте матрицу ошибок для модели Random Forest
plot_confusion_matrix_colorful(cm_rf, 'Random Forest')
```


    
![png](README_files/README_40_0.png)
    



    
![png](README_files/README_40_1.png)
    



    
![png](README_files/README_40_2.png)
    



```python
from sklearn.model_selection import GridSearchCV

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2']
}

# MultinomialNB не имеет множества гиперпараметров для настройки

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# объекты GridSearchCV для каждой модели
lr_grid_search = GridSearchCV(LogisticRegression(), lr_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Grid Search для каждой модели на обучающих данных
lr_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)

# наилучшие параметры для каждой модели
best_lr_params = lr_grid_search.best_params_
best_rf_params = rf_grid_search.best_params_

# производительность моделей с наилучшими параметрами на тестовых данных
best_lr_model = LogisticRegression(**best_lr_params)
best_rf_model = RandomForestClassifier(**best_rf_params)

best_lr_model.fit(X_train, y_train)
best_rf_model.fit(X_train, y_train)

y_pred_best_lr = best_lr_model.predict(X_test)
y_pred_best_rf = best_rf_model.predict(X_test)

# отчеты о классификации и точности для обновленных моделей
print("Лучшая модель Logistic Regression:")
print(classification_report(y_test, y_pred_best_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_best_lr)}\n")

print("Лучшая модель Random Forest:")
print(classification_report(y_test, y_pred_best_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_best_rf)}\n")

```

    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    

    C:\Users\ASUS\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Fitting 5 folds for each of 108 candidates, totalling 540 fits
    

    C:\Users\ASUS\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Лучшая модель Logistic Regression:
                  precision    recall  f1-score   support
    
               1       0.53      0.46      0.50       121
               2       0.34      0.15      0.20        96
               3       0.00      0.00      0.00        52
               4       0.50      0.06      0.11        62
               5       0.00      0.00      0.00        58
               6       0.16      0.05      0.07        64
               7       0.29      0.03      0.05        72
               8       0.21      0.14      0.17       167
               9       0.30      0.32      0.31       332
              10       0.51      0.81      0.62       607
    
        accuracy                           0.43      1631
       macro avg       0.28      0.20      0.20      1631
    weighted avg       0.37      0.43      0.37      1631
    
    Accuracy: 0.43041079092581236
    
    Лучшая модель Random Forest:
                  precision    recall  f1-score   support
    
               1       0.49      0.33      0.40       121
               2       0.35      0.14      0.20        96
               3       0.00      0.00      0.00        52
               4       0.50      0.06      0.11        62
               5       0.50      0.02      0.03        58
               6       0.00      0.00      0.00        64
               7       0.00      0.00      0.00        72
               8       0.25      0.09      0.13       167
               9       0.30      0.24      0.27       332
              10       0.47      0.89      0.61       607
    
        accuracy                           0.43      1631
       macro avg       0.29      0.18      0.17      1631
    weighted avg       0.35      0.43      0.34      1631
    
    Accuracy: 0.42611894543225015
    
    

![image.png](README_files/image.png)


```python
your_review_text = """
Самая УЖАСНАЯ поездка за всю мою жизнь! Пишу только сейчас, потому что деньги Амра Туристик за некачественно оказанные услуги не возвращает, на письменные претензии и звонки не реагирует )) А я пыталась получить возврат 2 недели.
5 августа я поехала в Кабардино-Балкарию - хотелось увидеть горы, старинные аулы и башни, которые были заявлены в программе.
50 человек повезли на старом автобусе, грязные сиденья в пятнах, сломанные ремни безопасности и практически неработающий кондиционер - то, что нужно за наши деньги!
В итоге при подъеме в горы в районе 13 часов дня автобус сломался. Как сказала гид - ну жарко же. А зачем на непригодной для подъема в горы технике везти людей? Мы простояли на обочине трассы в жару +38 почти 2 часа, пока сотрудник компании пыталась созвониться с Амра Туристик.
В итоге в горы мы не поднялись, никаких аулов и башен не увидели, спустились к озеру в кафе, пока нам продолжали искать автобус для возвращения в Краснодар. Ок, нашли автобус, позвали забирать вещи - внимание, и мы еще 2 часа просидели у обочины! Потому что автобус все еще «сейчас приедет».
Итого - я не увидела то, ради чего ехала 9 часов в одну сторону, заявленная программа не была выполнена со стороны туроператора, я провела 4 часа на обочине дороге - отличная экскурсия, Я ПЛАТИЛА НЕ ЗА ЭТО! Амра Туристик даже извинения не принесла, никакой компенсации предложено не было! И мои требования о возврате средств они игнорируют. При этом деньги принимают на карту физ лица (хотя это ООО) и наличкой в автобусе, чеки не выдают.
Кстати, в нашем автобусе были люди, которые должны были ехать в Адыгею, но им отменили поездку, т.к. сломался автобус. Видимо, этот автобус в итоге повез нас ????
Отвратительная компания, надеюсь, вы скоро закроетесь с учетом такого отношения к клиентам!
"""
```


```python
your_review_processed = preprocess_text(your_review_text)  # Здесь preprocess_text() - ваша функция предобработки
```


```python
# Векторизируйте ваш отзыв
your_review_vectorized = tfidf_vectorizer.transform([your_review_processed])
```


```python
# Предскажите оценку с помощью каждой модели
lr_prediction = lr_model.predict(your_review_vectorized)
nb_prediction = nb_model.predict(your_review_vectorized)
rf_prediction = rf_model.predict(your_review_vectorized)
```


```python
# Выведите предсказанные оценки
print(f"Предсказанная оценка (Logistic Regression): {lr_prediction[0]}")
print(f"Предсказанная оценка (Naive Bayes): {nb_prediction[0]}")
print(f"Предсказанная оценка (Random Forest): {rf_prediction[0]}")
```

    Предсказанная оценка (Logistic Regression): 1
    Предсказанная оценка (Naive Bayes): 1
    Предсказанная оценка (Random Forest): 1
    

![image.png](README_files/image.png)


```python
your_review_text = """
Я не знаю зачем тратить такие деньги на Гудзон, 
проходите мимо.. - Отзывы об Отель "GoodZone Club", 
Затока Слава богу были тут всего 3 дня
ни за что больше сюда не вернусь 
В отзыве еще не все написано про номер. Уже не хватило сил)
Из плюсов - басейн, джакузи,.. и пожалуй все
"""
```


```python
your_review_processed = preprocess_text(your_review_text)  # Здесь preprocess_text() - ваша функция предобработки
```


```python
# Векторизируйте ваш отзыв
your_review_vectorized = tfidf_vectorizer.transform([your_review_processed])
```


```python
# Предскажите оценку с помощью каждой модели
lr_prediction = lr_model.predict(your_review_vectorized)
nb_prediction = nb_model.predict(your_review_vectorized)
rf_prediction = rf_model.predict(your_review_vectorized)
```


```python
# Выведите предсказанные оценки
print(f"Предсказанная оценка (Logistic Regression): {lr_prediction[0]}")
print(f"Предсказанная оценка (Naive Bayes): {nb_prediction[0]}")
print(f"Предсказанная оценка (Random Forest): {rf_prediction[0]}")
```

    Предсказанная оценка (Logistic Regression): 2
    Предсказанная оценка (Naive Bayes): 9
    Предсказанная оценка (Random Forest): 1
    


```python

```
