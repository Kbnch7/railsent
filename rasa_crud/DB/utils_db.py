import sqlite3
import datetime
from DB import create
from DB import queries


def create_plans(user_id: str, place: str, time: str):
    """
    Функция, которая записывает определенный план (запись в таблицу
    "plans" базы данных) на указанное время.

    На вход подаются:
     - user_id (str)
     - place (str)
     - time (str)

    Возвращает True, если запись в таблицу "plans" базы данных была сделана.

    Возвращает False, если запись в таблицу "plans" базы данных не сделана
    (некорректные входные данные).
    """
    # Создает бд, если таковая отсутствует.
    create.start_database()

    # Если хоть один параметр для записи в таблицу "plans" базы данных
    # отсутствует, то функция вернет False.
    if user_id is None or place is None or time is None:
        return False
    else:
        conn: sqlite3.Connection = sqlite3.connect('database.db')
        with conn:
            cursor: sqlite3.Cursor = conn.cursor()

            # Запрос к таблице "plans" базы данных (вставка новой записи).
            sql_query: str = queries.insert_query
            arguments: tuple = (int(user_id), str(place), str(time))

            cursor.execute(sql_query, arguments)
            conn.commit()
            return True


def read_plans(user_time: str) -> list:
    """Функция, которая считывает планы (записи в таблице "plans" базы
    данных), записанные на указанное время (колонка "time" ) в базе данных.

    На вход подаются:
     - user_time (str)

    Возвращает все записи (list) с таблицы "plans" базы данных, в которых
    значение колонки "time" совпадает с входными данными.
    """
    # Создает бд, если таковая отсутствует.
    create.start_database()

    conn: sqlite3.Connection = sqlite3.connect('database.db')
    with conn:
        cursor: sqlite3.Cursor = conn.cursor()

        # Поиск времени (колонка "time"), начало которого совпадает с
        # указанным пользователем временем (его датой) в таблице "plans".
        sql_query: str = queries.read_query
        arguments: tuple = (str(user_time.split(' ')[0])+'%',)

        # Получение результата запроса.
        cursor.execute(sql_query, arguments)
        response: list = cursor.fetchall()

        answer: list = []

        # Если в указанном пользователем времени больше 1 параметра (например,
        # дата и время), то происходят дополнительные проверки.
        if len(user_time.split()) > 1:

            # Преобразования введенного пользователем времени к формату,
            # который используется в таблице "plans" колонке "time" базы
            # данных.
            user_time: datetime.datetime = datetime.datetime.strptime(
                user_time, '%Y-%m-%d %H:%M:%S'
                )

            # Цикл, который обрабатывает значения, полученные по запросу из
            # таблицы "plans" базы данных (переменная response).
            for plan in response:

                # Ожидание исключения необходимо для того случая, если время в
                # таблице "plans" колонке "time" записано не в том формате,
                # который ожидалось (или значение может быть None).
                try:
                    # Приведение времени к общему формату, полученного из
                    # колонки "time" таблицы "plans" базы данных.
                    plan_time: datetime.datetime = datetime.datetime.strptime(
                        plan[2], '%Y-%m-%d %H:%M:%S'
                        )

                    # Создание переменных, отвечающих за промежуток от
                    # введенного пользователем времени, в котором следует
                    # искать планы в колонке "time" таблице "plans" базы
                    # данных.
                    time_delta_before: datetime.timedelta = datetime.timedelta(
                        minutes=30
                        )
                    time_delta_after: datetime.timedelta = datetime.timedelta(
                        hours=1,
                        minutes=15
                        )

                    # Если +1.25 и -0.5 часа от введенного пользователем
                    # времени существует план, то он добавится в финальный
                    # ответ.
                    if (user_time - plan_time < time_delta_before and
                            plan_time - user_time < time_delta_after):
                        answer.append(plan)
                except Exception:
                    continue

            # Возврат финального ответа.
            return answer

        # Иначе, если параметров времени 1 или они вовсе отсутствуют, то
        # дополнительных проверок не требуется, результатом будет весь
        # результат запроса к базе данных (переменная "response").
        else:
            return response


def update_plans(place: str, old_time: str, new_time: str) -> bool:
    '''
    Функция, которая обновляет время (колонка "time") указанного плана
    (запись в таблице "plans" базы данных) по указанному месту
    (колонка "place").

    На вход подаются:
     - place (str)
     - old_time (str)
     - new_time (str)

    Возвращает значение True, если запись в базе данных обновлена.

    Возвращает значение False, если запись в базе данных не обновлена
    (запись в таблице "plans" базы данных по входным данным не найдена).
    '''
    # Создает бд, если таковая отсутствует.
    create.start_database()

    conn: sqlite3.Connection = sqlite3.connect('database.db')
    with conn:
        cursor: sqlite3.Cursor = conn.cursor()

        # Поиск записей, в которых время (колонка "time") совпадает с
        # указанным пользователем временем и место (колонка "place")
        # совпадает с указанным пользователем местом, в таблице "plans"
        # базы данных.
        sql_query: str = queries.check_existence_query
        arguments: tuple = (place, old_time)

        # Запись ответа в переменную "result".
        cursor.execute(sql_query, arguments)
        result: list = cursor.fetchall()

        # Если переменная "result" содержит данные (не пустая), то мы
        # обновляем время (колонка "time") в таблице "plans" базы данных,
        # где место (колонка "place") соответствует заданному пользователем
        # месту и возвращаем значение True.
        if result:
            sql_query: str = queries.update_query
            arguments: tuple = (new_time, old_time, place)
            cursor.execute(sql_query, arguments)
            return True

        # Иначе переменная "result" не содержит данные, значит у пользователя
        # нет записей с данными, которые он ввел, значит можно не обновлять
        # данные в таблице "plans" базы данных и вернуть значение False.
        else:
            return False


def delete_plans(place: str, time: str) -> bool:
    '''
    Функция, которая удаляет указанный план (запись в таблице базы данных)
    по времени (колонка "time") и месту (колонка "place") из таблицы базы
    данных "plans".

    На вход подаются:
     - place (str)
     - time (str)

    Возвращает True, если запись в таблице "plans" базы данных удалена.

    Возвращает False, если запись в таблице "plans" базы данных не удалена
    (запись не найдена).
    '''
    # Создает бд, если таковая отсутствует.
    create.start_database()

    conn: sqlite3.Connection = sqlite3.connect('database.db')
    with conn:
        # Если хоть один параметр для записи в таблицу "plans" базы данных
        # отсутствует, то функция вернет False.
        if place is None or time is None:
            return False
        else:
            # Поиск записей, в которых время (колонка "time") совпадает с
            # указанным пользователем временем и место (колонка "place")
            # совпадает с указанным пользователем местом, в таблице "plans"
            # базы данных.
            cursor: sqlite3.Cursor = conn.cursor()
            sql_query: str = queries.check_existence_query
            arguments: tuple = (place, time)

            # Запись результата запроса в переменную "result".
            cursor.execute(sql_query, arguments)
            result: list = cursor.fetchall()

            # Если такие записи существуют, то удаляем их и возвращаем True.
            if result:
                cursor: sqlite3.Cursor = conn.cursor()
                sql_query: str = queries.delete_query
                arguments: tuple = (place, time)
                cursor.execute(sql_query, arguments)
                return True
            # Иначе возвращаем False.
            else:
                return False
