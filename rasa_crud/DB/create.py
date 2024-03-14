import sqlite3


def start_database() -> None:
    """
    Функция для создания таблицы "plans" в базе данных, если она не существует.

    На вход не подаются данные.

    На выход не подаются данные
    """
    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS plans (user_id INTEGER, place TEXT, time TEXT );")
    conn.commit()

    cursor.close()
    conn.close()
