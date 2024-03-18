import sqlite3
from DB import queries


def start_database() -> None:
    """
    Функция для создания таблицы "plans" в базе данных, если она не существует.

    На вход не подаются данные.

    На выход не подаются данные
    """
    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute(queries.create_db_query)
    conn.commit()

    cursor.close()
    conn.close()
