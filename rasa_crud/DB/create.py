import sqlite3
from DB import queries


def start_database() -> None:
    """
    Функция для создания таблицы в базе данных, если она не существует.

    На вход не подаются данные.

    На выход не подаются данные
    """
    connection = sqlite3.connect("database.db")

    cursor = connection.cursor()

    cursor.execute(queries.create_db_query)
    connection.commit()

    cursor.close()
    connection.close()
