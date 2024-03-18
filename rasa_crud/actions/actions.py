from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import rutimeparser
import pymorphy2
from DB import utils_db


class CreateRow(Action):
    """Класс, который описывает функцию CREATE в архитектуре CRUD."""

    def name(self) -> Text:
        """
        Метод, который дает название этому действию,
        позволяя интегрировать его в rasa.
        """

        return "action_create_plans_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Метод, который отвечает за реализацию логики
        добавления записи в ежедневник.
        """

        # Инициализация переменных, в которых позже будут накапливаться данные.
        place: str = ''
        place_human: str = ''
        full_time_human: str = ''
        full_time: str = ''

        # Подготовка объектов, с помощью которых будут анализироваться
        # сущности из сообщения пользователя.
        morph: pymorphy2.MorphAnalyzer = pymorphy2.MorphAnalyzer()
        entities: list = tracker.latest_message.get('entities', [])

        # Перебор в цикле всех сущностей, которые распознались.
        for entity in entities:
            # Если сущность является "place", то накапливаем информацию в
            # переменных "place_human" и "place".

            # *place_human - переменная, в которой будет храниться информацию
            # о месте, которое необходимо записать в
            # ежедневник в том виде, в котором его сообщил пользователь.

            # *place - переменная, в которой будет храниться информация о
            # месте, которое необходимо записать в ежедневник, после приведения
            # к начальной форме с помощью лемматизации.

            if entity['entity'] == 'place':
                place_human += entity['value']
                place += str(morph.parse(entity['value'])[0].normal_form)

            # Иначе сущность является "time", то накапливаем информацию в
            # переменных "full_time_human" и "full_time".

            # *full_time_human - переменная, в которой будет храниться
            # информацию о времени, которое необходимо записать в ежедневник
            # после приведения к начальной форме с помощью лемматизации.

            # *full_time - переменная, в которой будет храниться информация
            # о времени, которое необходимо записать в ежедневник, после
            # приведения к начальной форме с помощью лемматизации и приведения
            # к общему виду базы данных с помощью rutimeparser.

            elif entity['entity'] == 'time':
                time_human: str = str(morph.parse(entity['value'])[0]
                                      .normal_form.lower())
                full_time_human += f"{time_human} "
                full_time: str = str(rutimeparser.parse(full_time_human))
        full_time_human: str = full_time_human[:-1]

        try:
            # Попытка сделать запись в ежедневник.
            query: bool = utils_db.create_plans(0, place, full_time)

            # Если попытка успешная - вывод сообщения пользователю об успешной
            # записи.
            if query:
                dispatcher.utter_message(text=f'Запись "{place_human}" '
                                         'записана в календарь на '
                                         f'{full_time_human}.')

            # Иначе вызов исключения.
            else:
                raise Exception

        # Если возникла ошибка по ходу выполнения или запись в ежедневник не
        # была успешной - вывод пользователю сообщения об этом.
        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []


class ReadRow(Action):
    """Класс, который описывает функцию READ в архитектуре CRUD."""

    def name(self) -> Text:
        """
        Метод, который дает название этому действию, позволяя интегрировать
        его в rasa.
        """

        return "action_read_plans_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Метод, который отвечает за реализацию логики чтения записи в
        ежедневник.
        """

        # Инициализация переменной, в которой позже будут накапливаться данные.
        full_entity: str = ''

        # Подготовка объектов, с помощью которых будут анализироваться сущности
        # из сообщения пользователя.
        entities: list = tracker.latest_message.get('entities', [])
        morph: pymorphy2.MorphAnalyzer = pymorphy2.MorphAnalyzer()

        # Перебор в цикле всех сущностей, которые распознались и их накопление
        # в переменной "full_entity".
        for entity in entities:
            full_entity += f'{entity["value"]} '
        full_entity: str = full_entity[:-1]

        # Преобразование данных из переменной full_entity

        # *time_human - переменная, в которой будет храниться информацию о
        # времени, по которому необходимо считать записи из ежедневника, после
        # приведения к начальной форме с помощью лемматизации.

        # *time - переменная, в которой будет храниться о времени, по которому
        # необходимо считать записи из ежедневника, после приведения к формату
        # хранения времени в базе данных.
        time_human: str = str(morph.parse(full_entity)[0].normal_form.lower())
        time: str = str(rutimeparser.parse(time_human))

        try:
            # Попытка прочитать данные из ежедневника.
            user_plans: list = utils_db.read_plans(time)

            # Если удалось прочитать данные из ежедневника, то вывожу
            # их пользователю.
            if user_plans:
                for plan in user_plans:
                    dispatcher.utter_message(text=f'На {time_human} '
                                             f'у вас "{plan[1]}". '
                                             f'Когда - {plan[2]}.')

            # Иначе сообщаю, что у пользователя нету данных на указанное
            # им время.
            else:
                dispatcher.utter_message(text='У вас нету планов'
                                         f'на {time_human}.')

        # Если произошла ошибка по мере выполнения, сообщаю об этом
        # пользователю.
        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []


class UpdateRow(Action):
    """Класс, который описывает функцию UPDATE в архитектуре CRUD."""

    def name(self) -> Text:
        """
        Метод, который дает название этому действию, позволяя интегрировать
        его в rasa.
        """

        return "action_update_plans_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Метод, который отвечает за реализацию логики обновления записей в
        ежедневнике.
        """

        # Инициализация переменных, в которых позже будут накапливаться данные.
        old_time: str = ''
        new_time: str = ''
        place: str = ''

        # Подготовка объектов, с помощью которых будут анализироваться сущности
        # из сообщения пользователя.
        entities: list = tracker.latest_message.get('entities', [])
        morph: pymorphy2.MorphAnalyzer = pymorphy2.MorphAnalyzer()

        # Перебор в цикле всех сущностей, которые распознались.
        for entity in entities:
            # Присвоение переменным "old_time", "new_time", "place"
            # соотвествующих сущностей (их названия).
            if entity['entity'] == "old_time":
                old_time: str = entity['value']
            elif entity['entity'] == "new_time":
                new_time: str = entity['value']
            elif entity['entity'] == "place":
                place: str = entity['value']

        # Преобразование данных из переменной "old_time".

        # *old_time_human - переменная, в которой будет храниться информацию о
        # времени, по которому необходимо считать записи из ежедневника и
        # заменить их на новые, после приведения к начальной форме с помощью
        # лемматизации.

        # *old_time - переменная, в которой будет храниться о времени, по
        # которому необходимо считать записи из ежедневника и заменить их на
        # новые, после приведения к формату хранения времени в базе данных.
        old_time_human: str = str(morph.parse(old_time)[0].normal_form.lower())
        old_time: str = str(rutimeparser.parse(old_time_human))

        # Преобразование данных из переменной "new_time".

        # *new_time_human - переменная, в которой будет храниться информацию о
        # времени, на которую необходимо заменить записи из ежедневника, после
        # приведения к начальной форме с помощью лемматизации.

        # *new_time - переменная, в которой будет храниться информацию о
        # времени, на которую необходимо заменить записи из ежедневника, после
        # приведения к начальной форме с помощью лемматизации.
        new_time_human: str = str(morph.parse(new_time)[0].normal_form.lower())
        new_time: str = str(rutimeparser.parse(new_time_human))

        # Преобразование данных из переменной "place"/
        place: str = str(morph.parse(place)[0].normal_form.lower())

        try:
            # Попытка обновить данные в ежедневнике.
            result: bool = utils_db.update_plans(place, old_time, new_time)

            # Если удалось обновить данные в ежедневнике, то вывожу сообщение
            # пользователю об этом.
            if result:
                dispatcher.utter_message(text=f'План "{place}" перенесен с '
                                         f'"{old_time}" на "{new_time}"')

            # Иначе сообщаю, что планов на указанное время нету в ежедневнике.
            else:
                dispatcher.utter_message(text='У вас не запланирован '
                                         f'"{place}" на время (дату) '
                                         f'"{old_time}"')

        except Exception:
            # При ошибке по ходу выполнения программы, вывод сообщения об этом
            # пользователю.
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []


class DeleteRow(Action):
    """Класс, который описывает функцию DELETE в архитектуре CRUD."""

    def name(self) -> Text:
        """
        Метод, который дает название этому действию, позволяя интегрировать
        его в rasa.
        """

        return "action_delete_plans_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Метод, который отвечает за реализацию логики удаления записей в
        ежедневнике.
        """

        # Инициализация переменных, в которых позже будут накапливаться данные.
        place: str = ''
        place_human: str = ''
        full_time_human: str = ''
        full_time: str = ''

        # Подготовка объектов, с помощью которых будут анализироваться сущности
        # из сообщения пользователя.
        entities: list = tracker.latest_message.get('entities', [])
        morph: pymorphy2.MorphAnalyzer = pymorphy2.MorphAnalyzer()

        # Перебор в цикле всех сущностей, которые распознались.
        for entity in entities:
            # Если сущность является "place", то накапливаем информацию в
            # переменных "place_human" и "place".

            # *place_human - переменная, в которой будет храниться информацию о
            # месте, запись о котором необходимо удалить из ежедневника, в том
            # виде, в котором его сообщил пользователь.

            # *place - переменная, в которой будет храниться информация о
            # месте, запись о котором необходимо удалить из ежедневника, после
            # приведения к начальной форме с помощью лемматизации.
            if entity['entity'] == 'place':
                place_human += entity['value']
                place += str(morph.parse(entity['value'])[0].normal_form)

            # Иначе сущность является "time", то накапливаем информацию в
            # переменных "full_time_human" и "full_time".

            # *full_time_human - переменная, в которой будет храниться
            # информацию о времени, запись о котором необходимо удалить из
            # ежедневника, после приведения к начальной форме с помощью
            # лемматизации.

            # *full_time - переменная, в которой будет храниться информация о
            # времени, запись о котором необходимо удалить из ежедневника,
            # после приведения к начальной форме с помощью лемматизации и
            # приведения к общему виду базы  данных с помощью rutimeparser.
            elif entity['entity'] == 'time':
                time_human: str = str(morph.parse(entity['value'])[0]
                                      .normal_form.lower())
                full_time_human += f"{time_human} "
                full_time: str = str(rutimeparser.parse(full_time_human))
        full_time_human: str = full_time_human[:-1]

        try:
            # Попытка удалить запись в ежедневнике.
            result: bool = utils_db.delete_plans(place, full_time)

            # Если удалось удалить данные с ежедневника, то вывожу сообщние
            # пользователю об этом.
            if result:
                dispatcher.utter_message(text=f'Ваш план "{place_human}" на '
                                         f'время (дату) "{full_time_human}" '
                                         'удален.')

            # Иначе сообщаю, что планов на указанное время нету в ежедневнике.
            else:
                dispatcher.utter_message(text='У вас нету плана на время, '
                                         'которое вы указали.')

        # При ошибке по ходу выполнения программы, вывод сообщения об этом
        # пользователю.
        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []
