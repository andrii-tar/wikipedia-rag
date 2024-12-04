## Project Title: Wikipedia Data RAG Lab 7

## Задача:
Розробка системи для ефективного пошуку та ранжування статей з використанням великих мовних моделей (LLM), ретериверів, перерейтингів та покращення UX через інтерфейс Gradio.

## Короткий опис компонентів:

### 1. Data Source
- Використано архів статей з Вікіпедії. Дані з [Kaggle: Wikipedia Data](https://www.kaggle.com/datasets/ismaeldwikat/wikipedia/data)

### 2. Chunking
- Власна функція розбиття на однакові блоки з перекриттям 10%

### 3. LLM
  - Використано модель llama3-8b-8192 з сервісу Groq

### 4. Retriever
Реалізовано два варіанти для пошуку відповідної інформації:
  - Семантичний пошук за допомогою SentenceTransformer
  - Пошук за ключовими словами з BM25

Можна використати один з зазначених методів або вимкнути додаткову інформацію з RAG для порівняння результатів

### 5. Reranker
- N/A

### 6. Citations
  - Для кожної відповіді доступна інформація про **chunk_id**, **article_id** за якими можна перевірити джерела інформації. Наприклад оримати **назву статті та посилання**

### 7. UI
- Gradio

### 8. Other
- N/A

### Участь у проєкті:
- **Тарасов Андрій КН-415**
- **Павлів Дмитро  КН-417**

### Посилання на запущений сервіс:
- Сервіс розгортається локально у **Docker** з репозиторію.
- **Посилання на source code:**
https://github.com/andrii-tar/wikipedia-rag.git

## Інструкції з запуску:
1. Для розгортання проєкту, клонувати репозиторій:
   ```bash
   git clone https://github.com/andrii-tar/wikipedia-rag.git
   
2. Виконати команду для запуску з Docker:

    ``` bash
    docker-compose up --build
    ```

Після цього додаток буде доступний на http://localhost:7860.


## Приклади запитів:

1. Where Meir Shalev was born?
2. Who was the banker of Barclays Bank in the 19th century?
3. How many Afghani Sikhs were in Talta's book which was published in 2014?
4. How many Assamese Sikhs were in Assam at the beginning of the 21st century?

**Система надає повну відповідь на зазначені питання, не залежно від обраного ретрівера. Натомість, якщо виключити RAG знань моделі не вистачає.**

Повні тексти використаних статей знаходяться у файлі dataframe.csv
