from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import rutimeparser
import pymorphy2
from DB import utils_db


class CreateRow(Action):
    """CREATE."""

    def name(self) -> Text:
        return "action_create_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Добавление записи в ежедневник.
        """

        place = ''
        place_from_user = ''
        full_time_from_user = []
        full_time = ''

        # Анализатор, с помощью которого будет определяться начальная форма.
        morph = pymorphy2.MorphAnalyzer()

        # Сущности из последнего сообщения пользователя.
        entities = tracker.latest_message.get('entities', [])

        for entity in entities:

            # place_from_user - место, которое необходимо записать в
            # ежедневник в том виде, в котором его сообщил пользователь.

            # place - место, которое необходимо записать в ежедневник, после приведения
            # к начальной форме с помощью лемматизации.

            if entity['entity'] == 'place':
                place_from_user += entity['value']

                # Индекс 0 для выбора наиболее вероятного распознавания.
                place += morph.parse(entity['value'])[0].normal_form

            # full_time_from_user - время, которое необходимо записать в ежедневник
            # после приведения к начальной форме с помощью лемматизации.

            # full_time - время, которое необходимо записать в ежедневник, после
            # приведения к начальной форме с помощью лемматизации и приведения
            # к общему виду базы данных с помощью rutimeparser.

            elif entity['entity'] == 'time':
                # Индекс 0 для выбора наиболее вероятного распознавания.
                full_time_from_user.append(morph.parse(entity['value'])[0].normal_form.lower())

        full_time_from_user = " ".join(full_time_from_user)
        full_time = str(rutimeparser.parse(full_time_from_user))

        try:
            query = utils_db.create_plans(0, place, full_time)

            if query:
                dispatcher.utter_message(text=f'Запись "{place_from_user}" '
                                         'записана в календарь на '
                                         f'{full_time_from_user}.')

            else:
                raise Exception

        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []


class ReadRow(Action):
    """READ."""

    def name(self) -> Text:
        return "action_read_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Чтение записи из ежедневника.
        """

        full_entity = []

        # Анализатор, с помощью которого будет определяться начальная форма.
        morph = pymorphy2.MorphAnalyzer()

        # Сущности из последнего сообщения пользователя.
        entities = tracker.latest_message.get('entities', [])

        for entity in entities:
            if entity["entity"] == "time":
                full_entity.append(entity["value"])

        full_entity = " ".join(full_entity)

        # time_from_user - время, по которому необходимо считать записи из
        # ежедневника, после приведения к начальной форме с помощью лемматизации.
        # Индекс 0 для выбора наиболее вероятного распознавания.
        time_from_user = morph.parse(full_entity)[0].normal_form.lower()

        # time - время, по которому необходимо считать записи из ежедневника,
        # после приведения к формату хранения времени в базе данных.
        time = str(rutimeparser.parse(time_from_user))

        try:
            user_plans = utils_db.read_plans(time)

            if user_plans:
                for plan in user_plans:
                    dispatcher.utter_message(text=f'На {time_from_user} '
                                             f'у вас "{plan[1]}". '
                                             f'Когда - {plan[2]}.')

            else:
                dispatcher.utter_message(text='У вас нету планов '
                                         f'на {time_from_user}.')

        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []


class UpdateRow(Action):
    """UPDATE."""

    def name(self) -> Text:
        return "action_update_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Обновления записи в ежедневнике.
        """

        old_time = ''
        new_time = ''
        place = ''

        # Анализатор, с помощью которого будет определяться начальная форма.
        morph = pymorphy2.MorphAnalyzer()

        # Сущности из последнего сообщения пользователя.
        entities = tracker.latest_message.get('entities', [])

        for entity in entities:
            if entity['entity'] == "old_time":
                old_time = entity['value']
            elif entity['entity'] == "new_time":
                new_time = entity['value']
            elif entity['entity'] == "place":
                place = entity['value']

        # old_time_from_user - время, по которому необходимо считать записи
        # из ежедневника и заменить их на новые, после приведения к
        # начальной форме с помощью лемматизации.
        # Индекс 0 для выбора наиболее вероятного распознавания.
        old_time_from_user = morph.parse(old_time)[0].normal_form.lower()

        # old_time - время, по которому необходимо считать записи из
        # ежедневника и заменить их на новые, после приведения к формату
        # хранения времени в базе данных.
        old_time = str(rutimeparser.parse(old_time_from_user))

        # new_time_from_user - время, на которое необходимо заменить записи
        # из ежедневника, после приведения к начальной форме с помощью
        # лемматизации.
        # Индекс 0 для выбора наиболее вероятного распознавания.
        new_time_from_user = morph.parse(new_time)[0].normal_form.lower()

        # new_time - время, на которое необходимо заменить записи из
        # ежедневника, после приведения к начальной форме с помощью
        # лемматизации.
        new_time = str(rutimeparser.parse(new_time_from_user))

        # Индекс 0 для выбора наиболее вероятного распознавания.
        place = morph.parse(place)[0].normal_form.lower()

        try:
            result = utils_db.update_plans(place, old_time, new_time)

            if result:
                dispatcher.utter_message(text=f'План "{place}" перенесен с '
                                         f'"{old_time}" на "{new_time}"')

            else:
                dispatcher.utter_message(text='У вас не запланирован '
                                         f'"{place}" на время (дату) '
                                         f'"{old_time}"')

        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []


class DeleteRow(Action):
    """DELETE."""

    def name(self) -> Text:
        return "action_delete_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Удаление записи из ежедневника.
        """

        place = []
        place_from_user = []
        full_time_from_user = []
        full_time = []

        # Анализатор, с помощью которого будет определяться начальная форма.
        morph = pymorphy2.MorphAnalyzer()

        # Сущности из последнего сообщения пользователя.
        entities = tracker.latest_message.get('entities', [])

        for entity in entities:

            # place_from_user - место, запись о котором необходимо удалить из
            # ежедневника, в том виде, в котором его сообщил пользователь.

            # place - место, запись о котором необходимо удалить из
            # ежедневника, после приведения к начальной форме с помощью
            # лемматизации.
            if entity['entity'] == 'place':
                place_from_user.append(entity['value'])

                # Индекс 0 для выбора наиболее вероятного распознавания.
                place.append(morph.parse(entity['value'])[0].normal_form)

            # full_time_from_user - время, запись о котором необходимо удалить
            # из ежедневника, после приведения к начальной форме с помощью
            # лемматизации.

            # full_time - время, запись о котором необходимо удалить из
            # ежедневника, после приведения к начальной форме с помощью
            # лемматизации и приведения к общему виду базы  данных с
            # помощью rutimeparser.
            elif entity['entity'] == 'time':
                # Индекс 0 для выбора наиболее вероятного распознавания.
                full_time_from_user.append(morph.parse(entity['value'])[0].normal_form.lower())
                full_time.append(str(rutimeparser.parse(full_time_from_user)))

        place_from_user = " ".join(place_from_user)
        place = " ".join(place)
        full_time = " ".join(full_time)
        full_time_from_user = " ".join(full_time_from_user)

        try:
            result = utils_db.delete_plans(place, full_time)

            if result:
                dispatcher.utter_message(text=f'Ваш план "{place_from_user}" на '
                                         f'время (дату) "{full_time_from_user}" '
                                         'удален.')

            else:
                dispatcher.utter_message(text='У вас нету плана на время, '
                                         'которое вы указали.')

        except Exception:
            dispatcher.utter_message(text='Возникла неожиданная ошибка, '
                                     'повторите попытку.')
        return []
