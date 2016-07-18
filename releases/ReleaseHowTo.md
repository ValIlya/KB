# Релизы

##1 релиз: Только средние и медианы за предыдущие 5 недель

Сборка: 
* [data_split_by_state.py](../scripts/data_split_by_state.py) - раскидывает трейн на штаты
* [preprocessing_all_data_v01.py](release_v01/preprocessing_all_data_v01.py) - собирает медианы и средние и сохраняем
* [models_timeseries_release_v01.py](release_v01/models_timeseries_release_v01.py) - приклеиваем тест-сет и считаем лаги, на таком датасете фитим модель

Качество на лидерборде: 0.55233

Адекватная оценка ошибки получается при прогнозе на 8 и 9 неделю: [models_timeseries_release_v01_2.py](release_v01/models_timeseries_release_v01_2.py)
