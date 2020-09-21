# internship_test


## Задание №1
task1.py -- решение задачи

## Задание №2

### Подготовка данных
Датасет разбиваю на тренировочную и валидационную выборки. Изображения обеих выборок привожу к единому размеру и нормирую. К изображениям тренировочной выборки применяю дополнительные аугментации.

### Процесс тренировки
Использую предобученную ResNet-18. Так как решается задача бинарной классификации и классы в датасете сбалансированы, то в качестве функции потерь использую кросс-энтропию. Функцию оптимизации выбрал Adam c learning_rate = 0.001. Обучение проводил в течение 15 эпох. 

### Результат
На валидационной выборке была достигнута точность 98.3%.

### Запуск тренировки
Тренировка проводилась в notebook-файле train.ipynb. Для тренировки нужно в первой клетке указать директорию с тренировочными данными

### Запуск нейросети
Запуск нейросети проводится с помощью скрипта process.py. 

Пример запуска: python3 process.py -t folder/to/process/
