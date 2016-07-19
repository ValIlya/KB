# Релизы

##1 релиз: Только средние и медианы за предыдущие 5 недель

Сборка: 
* [data_split_by_state.py](../scripts/data_split_by_state.py) - раскидывает трейн на штаты
* [preprocessing_all_data_v01.py](release_v01/preprocessing_all_data_v01.py) - собирает медианы и средние и сохраняем
* [models_timeseries_release_v01.py](release_v01/models_timeseries_release_v01.py) - приклеиваем тест-сет и считаем лаги, на таком датасете фитим модель

Качество на лидерборде: 0.55233

Адекватная оценка ошибки получается при прогнозе на 8 и 9 неделю: 
[models_timeseries_release_v01_2.ipynb](../models/models_timeseries_release_v01_2.ipynb)

##2 релиз: Средние за предыдущие 5 недель, средние по нескольким неделям сразу (wide lags), новые сплиты

* [preprocessing_all_data_v02.py](release_v02/preprocessing_all_data_v02.py) - собирает средние логарифмов, приклеивает тест-сет, лаги и сохраняем
* [preprocessing_all_data_v02_2.py](release_v02/preprocessing_all_data_v02_2.py) - меньше лагов и сплитов, оставлены самые важные
* [rf_feature_importance_v02.ipynb](release_v02/rf_feature_importance_v02.ipynb) - оценка качества фичей
