# ReadMe по этому кабаку

## Релизы

[Описание](releases/ReleaseHowTo.md)

## Скрипты

0. [preprocessing_time_series.py](scripts/preprocessing_time_series.py) - старый-добрый скрипт, где впервые приклеены все справочники и нагерены переменные по продукту + лаги по объемам и стоимости продаж
0. [crossval.py](scripts/crossval.py) - устаревший файлик с зафиксированной кросс-валидацией (которая не соотносится с лидербордом)
0. [gridsearch.py](scripts/gridsearch.py) - переборщик параметров для xgboost (и в принципе, при допиливании, для любой другой модельки)
0. [**data_split_by_state.py**](scripts/data_split_by_state.py) - очень важный скрипт, который раскидывает трейн на штаты (нужен для сборки любого релиза)

## Ноутбуки

0. [template.ipynb](template.ipynb) - базовый ноутбук с примером подхвата данных
0. [data_exploration.ipynb](data_exploration/data_exploration.ipynb) - графики, которые первые пришли в голову при обзоре переменных
0. [product_clustering.ipynb](data_exploration/product_clustering.ipynb) - кластеризация продуктов
0. [xgboost_v2.ipynb](models/product_clustering.ipynb) - xgboost и подбор параметров
0. [**feature_test.ipynb**](releases/release_v03/feature_test.ipynb) - пример вызова генератора фич прямо из ноутбука и тут же построение и валидация модельки (можно использовать его для проверки своих гипотез)
