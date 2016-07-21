# ReadMe по этому кабаку

## Релизы

[Описание](releases/ReleaseHowTo.md)

## Скрипты

0. preprocessing.py - приклеены все справочники и нагерены переменные по продукту
0. preprocessing_time_series.py - предыдущее + лаги по объемам и стоимости продаж
0. crossval.py - файлик с зафиксированной кросс-валидацией

## Ноутбуки

0. template.ipynb - базовый ноутбук с примером подхвата данных
0. data_exploration.ipynb - графики, которые первые пришли в голову при обзоре переменных
0. models_timeseries.ipynb - моделька с использованием лагов
0. product_clustering.ipynb - кластеризация продуктов
0. [how_to_find_strong_features.ipynb](how_to_find_strong_features.ipynb) - поиск хороших сплитов
0. [how_to_preprocess_only_strong_features.ipynb](how_to_preprocess_only_strong_features.ipynb) -  отбор трансофрмаций для генерации фич по всем штатам