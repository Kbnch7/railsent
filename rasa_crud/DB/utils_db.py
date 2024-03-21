import sqlite3
import datetime
from DB import queries


def start_database() -> None:
    """
    Создание таблицы в базе данных, если она не существует.
    """
    connection = sqlite3.connect("database.db")

    cursor = connection.cursor()

    cursor.execute(queries.create_db_query)
    connection.commit()

    cursor.close()
    connection.close()

def create_plans(user_id: str, place: str, time: str):
    """
    Запись определенного плана (запись в таблице
    базы данных) на указанное время.

    На вход подаются:
     - user_id (str) Идентификатор пользователя.
     - place (str) Место для записи в ежедневник.
     - time (str) Время для записи в ежедневник.

    Возвращает True, если запись в таблицу "plans" базы данных была сделана.

    Возвращает False, если запись в таблицу "plans" базы данных не сделана
    (некорректные входные данные).
    """
    # Создает бд, если таковая отсутствует.
    start_database()

    if user_id is None or place is None or time is None:
        return False

    connection = sqlite3.connect('database.db')
    with connection:
        cursor = connection.cursor()

        # Запрос к таблице "plans" базы данных (вставка новой записи).
        sql_query = queries.insert_query
        arguments = (user_id, place, time)

        cursor.execute(sql_query, arguments)
        connection.commit()
        return True


def read_plans(time: str) -> list:
    """
    Считывание планов (записи в таблице базы
    данных), записанных на указанное время (колонка "time" )
    в базе данных.

    На вход подаются:
     - user_time (str) Время, запись на которое необходимо считать.

    Возвращает все записи (list) с таблицы базы данных, в которых
    значение колонки "time" совпадает с входными данными.
    """
    # Создает бд, если таковая отсутствует.
    start_database()

    connection = sqlite3.connect('database.db')
    with connection:
        cursor = connection.cursor()

        # Поиск времени (колонка "time"), начало которого совпадает с
        # указанным пользователем временем (его датой) в таблице базы данных.
        sql_query = queries.read_query
        arguments = (time.split(' ')[0]+'%',)

        # Получение результата запроса.
        cursor.execute(sql_query, arguments)
        all_answer = cursor.fetchall()

        answer_by_time = []

        # Если в указанном пользователем времени больше 1 параметра (например,
        # дата и время), то планы будут браться в определенном промежутке.
        if len(time.split()) > 1:

            time = datetime.datetime.strptime(
                time, '%Y-%m-%d %H:%M:%S'
                )

            for plan in all_answer:

                # Ожидание исключения необходимо для того случая, если время в
                # таблице "plans" колонке "time" записано не в том формате,
                # который ожидалось (или значение может быть None).
                try:
                    plan_time = datetime.datetime.strptime(
                        plan[2], '%Y-%m-%d %H:%M:%S'
                        )

                    time_delta_before = datetime.timedelta(
                        minutes=30
                        )
                    time_delta_after = datetime.timedelta(
                        hours=1,
                        minutes=15
                        )

                    # Если +1.25 и -0.5 часа от введенного пользователем
                    # времени существует план, то он добавится в финальный
                    # ответ.
                    if (plan_time - time_delta_after < time <
                            plan_time + time_delta_before):
                        answer_by_time.append(plan)
                except Exception:
                    continue

            return answer_by_time

        # Иначе, если параметров времени 1 или они вовсе отсутствуют, то
        # дополнительных проверок не требуется, результатом будет весь
        # результат запроса к базе данных (переменная "response").
        return all_answer


def update_plans(place: str, old_time: str, new_time: str) -> bool:
    '''
    Обновление времени (колонка "time") указанного плана
    (запись в таблице "plans" базы данных) по указанному месту
    (колонка "place").

    На вход подаются:
     - place (str) Место, на которое в данный момент записан план.
     - old_time (str) Время, на которое в данный момент записан план.
     - new_time (str) Время, на которое необходимо перенести план.

    Возвращает значение True, если запись в базе данных обновлена.

    Возвращает значение False, если запись в базе данных не обновлена
    (запись в таблице "plans" базы данных по входным данным не найдена).
    '''
    # Создает бд, если таковая отсутствует.
    start_database()

    connection = sqlite3.connect('database.db')
    with connection:
        cursor = connection.cursor()

        # Поиск записей, в которых время (колонка "time") совпадает с
        # указанным пользователем временем и место (колонка "place")
        # совпадает с указанным пользователем местом, в таблице "plans"
        # базы данных.
        sql_query = queries.check_existence_query
        arguments = (place, old_time)

        cursor.execute(sql_query, arguments)
        result = cursor.fetchall()

        if result:
            sql_query = queries.update_query
            arguments = (new_time, old_time, place)
            cursor.execute(sql_query, arguments)
            return True
        return False


def delete_plans(place: str, time: str) -> bool:
    '''
    Удаляение указанного плана (запись в таблице базы данных)
    по времени (колонка "time") и месту (колонка "place") из таблицы базы
    данных "plans".

    На вход подаются:
     - place (str) Место, запись на которое необходимо удалить.
     - time (str) Время, запись на которое необходимо удалить.

    Возвращает True, если запись в таблице "plans" базы данных удалена.

    Возвращает False, если запись в таблице "plans" базы данных не удалена
    (запись не найдена).
    '''
    # Создает бд, если таковая отсутствует.
    start_database()

    conn = sqlite3.connect('database.db')
    with conn:
        # Если хоть один параметр для записи в таблицу "plans" базы данных
        # отсутствует, то функция вернет False.
        if place is None or time is None:
            return False
        # Поиск записей, в которых время (колонка "time") совпадает с
        # указанным пользователем временем и место (колонка "place")
        # совпадает с указанным пользователем местом, в таблице "plans"
        # базы данных.
        cursor = conn.cursor()
        sql_query = queries.check_existence_query
        arguments = (place, time)

        cursor.execute(sql_query, arguments)
        result = cursor.fetchall()

        if result:
            cursor = conn.cursor()
            sql_query = queries.delete_query
            arguments = (place, time)
            cursor.execute(sql_query, arguments)
            return True
        return False
