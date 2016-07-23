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

* [preprocessing_all_data_v03_1.py](release_v03/preprocessing_all_data_v02.py) - много сплитов и лагов - **для проверки новых фич изменяйте его**
* [feature_test.ipynb](release_v03/feature_test.ipynb) - пример расчета фич прямо в ноутбуке и валидация модельки на них

## Общий вид решения

0. Тренировочный сет разбит на штаты  [data_split_by_state.py](../scripts/data_split_by_state.py)
0. В итерации по каждому штату генерируются фичи [preprocessing_all_data_v03_1.py](release_v03/preprocessing_all_data_v02.py):
    0. присоединяются данные по городам (функция town_preproc)
    0. тестовый сет за нужные штаты присоединяется к тренировочному
    0. присоединяются данные по продуктам (products_preproc)
    0. считаются производные фичи от объемов: логарифмы, средние цены (volumes_preproc) 
    0. считаются переменные на сплитах: средние, минимумы, максимумы (splits_compute)
    0. считаются лаги по батчам, отдельным наборам продуктов, чтобы было быстрее (lag_batch_generation):
        0. лаги за 1 неделю: предыдущую, через одну, ... (lag_generation)
        0. "широкие" лаги за n предыдущих недель: средние за 3-6 предыдущих, без текущей недели (wide_lag_generation)
    0. на всякий случай кодировка во всех текстовых полях правится на несоответствие ASCII и убираются всякие умляуты (text_encoding)
    
0. Качество по отдельно взятому штату можно посмотреть так: [feature_test.ipynb](release_v03/feature_test.ipynb)
0. По каждому штату строится бустинг деревьями глубиной 10-12 деревьев [models_xgboost_release_v02.py](release_v02/models_xgboost_release_v02.py)

