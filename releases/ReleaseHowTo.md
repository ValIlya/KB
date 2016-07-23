# Релизы

Для каждого релиза используется набор данных по штатам:
* [data_split_by_state.py](../scripts/data_split_by_state.py) - раскидывает трейн на штаты

##1 релиз: 
Только средние и медианы за предыдущие 5 недель

Сборка: 
* [preprocessing_all_data_v01.py](release_v01/preprocessing_all_data_v01.py) - собирает медианы и средние и сохраняем
* [models_timeseries_release_v01.py](release_v01/models_timeseries_release_v01.py) - приклеиваем тест-сет и считаем лаги, на таком датасете фитим модель

Качество на лидерборде: 0.55233

Адекватная оценка ошибки получается при прогнозе на 8 и 9 неделю: 
[models_timeseries_release_v01_2.ipynb](../models/models_timeseries_release_v01_2.ipynb)

##2 релиз: 
Средние за предыдущие 5 недель, средние по нескольким неделям сразу (wide lags), новые сплиты

* [preprocessing_all_data_v02.py](release_v02/preprocessing_all_data_v02.py) - собирает средние логарифмов, приклеивает тест-сет, лаги и сохраняем
* [preprocessing_all_data_v02_2.py](release_v02/preprocessing_all_data_v02_2.py) - меньше лагов и сплитов, оставлены самые важные
* [preprocessing_all_data_v02_prod.py](release_v02/preprocessing_all_data_v02_prod.py) - генерация preprocessing_all_data_v02_2.py с оптимизацией
* [models_xgboost_release_v02.py](release_v02/models_xgboost_release_v02.py) - готовый скоринг по фичам из preprocessing_all_data_v02_prod.py
* [rf_feature_importance_v02.ipynb](release_v02/rf_feature_importance_v02.ipynb) - оценка качества фичей


Задел на будущее: Скрипт + исследование по важности сплитов и лагов:
* [preprocessing_all_data_v02_3.py](release_v02/preprocessing_all_data_v02_3.py) - выкидываем слабые переменные, **изменено название средних за n последние недель** - с заделом на новую переменную "последнее известное"
* [how_to_find_strong_features.ipynb](release_v02/how_to_find_strong_features.ipynb) - поиск хороших сплитов

##3 релиз: 
Гипотезы и не перечислить вот так: https://trello.com/c/OEEvMu6R

* [preprocessing_all_data_v03_1.py](release_v03/preprocessing_all_data_v02.py) - пока что повторяет продакшн 2 релиза с незначительными изменениями - **для проверки новых фич изменяйте его**
* [feature_test.ipynb](release_v03/feature_test.ipynb) - пример расчета фич прямо в ноутбуке и валидация модельки на них
