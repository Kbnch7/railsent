
Этот проект выполняет функцию ежедневника, позволяя пользователю записывать, считывать,
обновлять и удалять планы (время и место), управляя сервисом с помощью естественной речи.
***
Как развернуть сервис?
1. Скачать zip версию проекта.
2. Распаковать.
3. Установить Python3.9 (https://www.microsoft.com/store/productId/9P7QFQMJRFP7?ocid=pdpshare).
4. Выбрать интерпретатор Python3.9 (путь: C:\Users\*пользователь*\AppData\Local\Microsoft\WindowsApps\python3.9.exe )
5. Создать виртуальное окружение командой "python -m venv venv".
6. Активировать виртуальное окружение командой ".\venv\Scripts\activate" (windows), "source env/bin/activate" (linux).
7. Установить зависимости проекта командой "pip install -r requirements.txt" из папки с проектом.
8. Обучить модель с помощью команды "rasa train".
9. После обучения запустить модель командой "rasa shell".
10. Открыть еще один терминал, перейти в папку с проектом и активировать виртуальное окружение командой ".\venv\Scripts\activate"
11. Запустить custom actions, введя команду "rasa run actions" в 2 терминале.
12. Использовать бота, активируя один из 4 навыков (запись, чтение, обновление и удаление данных из ежедневника), в 1 терминале.
***
Что можно улучшить?
1. Перейти с sqlite3 на mysql/postgres.
2. Добавить custom exceptions для более точной реализации навыков CRUD (более точного вывода информации об ошибках).
3. Добавить больше примеров для каждого intent в целях улучшения распознавания.
***
Пример работы проекта:
1. Запись в ежедневник.

![create_function](./crud_functions_pictures/create.png)
2. Чтение из ежедневника.

![read_function](./crud_functions_pictures/read.png)
3. Обновление в ежедневнике.

![update_function](./crud_functions_pictures/update.png)
4. Удаление из ежедневника.

![delete_function](./crud_functions_pictures/delete.png)
