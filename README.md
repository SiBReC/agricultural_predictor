Продвинутая система прогнозирования урожайности сельскохозяйственных культур с использованием методов машинного обучения и нейронных сетей. Система анализирует многолетние данные урожайности, погодные условия, экономические показатели и рыночные индексы для создания точных прогнозов.

Основные возможности
1. Интерактивная картографическая система
Визуализация регионов Российской Федерации с границами
Выбор регионов непосредственно на карте
Отображение маркеров и подсветка выбранных регионов
Интеграция с OpenStreetMap

2. Комплексная загрузка и обработка данных
Данные урожайности по регионам и годам
Метеорологические данные (температура, давление, влажность)
Финансовые показатели (курсы валют USD/RUB)
Рыночные данные (цены на нефть)
Фондовый индекс Московской биржи
Геоданные (границы регионов в формате JSON)

3. Модели машинного обучения
Random Forest (ансамбль решающих деревьев)
XGBoost (градиентный бустинг)
Gradient Boosting
LSTM (долгая краткосрочная память для временных рядов)
Ансамблевые методы прогнозирования

4. Функционал прогнозирования
Поддержка различных сельскохозяйственных культур
Гибкая настройка периода прогноза (от 1 месяца до 5 лет)
Учет множественных факторов влияния
Расчет статистических показателей и доверительных интервалов
Оценка вероятности правильности прогноза

5. Визуализация и анализ результатов
Графики динамики урожайности
Сравнительный анализ с предыдущими периодами
Анализ важности факторов влияния
Вероятностные распределения прогнозо
Генерация аналитических отчетов

7. Экспорт данных
Сохранение полных отчетов в формате Excel
Экспорт графических материалов в формате PNG
Детализированные аналитические отчеты
Технологический стек

Интерфейс: Tkinter для графического интерфейса
Обработка данных: Pandas, NumPy
Визуализация: Matplotlib, Seaborn
Машинное обучение: Scikit-learn, XGBoost, LightGBM
Нейронные сети: TensorFlow/Keras
Картография: TkinterMapView
Форматы данных: Excel, JSON

Структура проекта
text
agricultural_predictor/
│
├── agricultural_data/          # Каталог с исходными данными
│   ├── Урожайность.xlsx       # Данные по урожайности
│   ├── Доллар пример.xlsx     # Данные по курсам валют
│   ├── oil_prices.xlsx        # Данные по ценам на нефть
│   ├── Индекс МосБиржи.xlsx   # Данные фондового индекса
│   ├── weather_data/          # Метеорологические данные
│   └── regions_borders.json   # Геоданные границ регионов
│
├── main.py                    # Основной исполняемый файл
├── requirements.txt           # Зависимости проекта
└── README.md                  # Документация

Установка и запуск
Предварительные требования
Python 3.8 или выше
8 ГБ оперативной памяти (рекомендуется)
2 ГБ свободного дискового пространства
Установка зависимостей
bash
pip install -r requirements.txt

Основные зависимости
txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
tensorflow>=2.11.0
tkintermapview>=1.17
openpyxl>=3.0.0

Запуск приложения
bash
python main.py

Форматы данных
Данные урожайности
Формат: Excel (.xlsx)

Лист: 'Лист1'
Структура:
Первый столбец: названия регионов
Последующие столбцы: годы (в качестве заголовков)
Ячейки: значения урожайности в ц/га

Метеорологические данные
Формат: Excel (.xlsx)
Размещение: каталог weather_data/
Обязательные поля: температура, давление, влажность, дата
Поддерживаемые наименования колонок:
Температура: 'temperature', 'temp'
Давление: 'pressure', 'davlenie'
Влажность: 'humidity', 'vlaga'
Дата: 'date', 'data', 'время', 'дата'

Экономические данные
Курсы валют: дата и значение курса
Цены на нефть: дата и цена
Индекс Московской биржи: дата и значение индекса
Геоданные
Формат: JSON
Структура: координаты полигонов границ регионов
Система координат: географические координаты (широта, долгота)

Руководство пользователя
1. Инициализация системы
Запустите приложение

Дождитесь автоматической загрузки данных
При необходимости загрузите границы регионов вручную

2. Подготовка моделей
Нажмите "Обучить модели" для инициализации алгоритмов машинного обучения
Система автоматически выберет оптимальные регионы для обучения
Отслеживайте процесс обучения в информационной панели

3. Работа с картой
Используйте карту Российской Федерации для выбора региона
Кликните на интересующий регион для его выделения
Используйте инструменты масштабирования и перемещения карты

4. Настройка параметров прогноза
Выберите сельскохозяйственную культуру из доступного списка
Установите период прогнозирования
Выберите модель прогнозирования

Настройте дополнительные параметры:
Учет погодных данных
Учет экономических показателей
Использование исторических данных
Длина исторического периода

5. Получение и анализ результатов
Просмотрите прогнозируемое значение урожайности
Изучите доверительный интервал и вероятность правильности
Проанализируйте динамику урожайности на графиках
Ознакомьтесь с важностью факторов влияния
Используйте сгенерированные рекомендации

6. Экспорт результатов
Сохраните полный отчет в формате Excel
Экспортируйте графики в формате изображений
Используйте аналитические отчеты для дальнейшего анализа

Архитектура системы
Модули системы
Config - конфигурационные параметры системы
AgriculturalDataLoader - загрузчик и процессор данных
AdvancedYieldPredictor - модуль машинного обучения и прогнозирования
MapHandler - обработчик картографических операций
PredictionDialog - диалоговое окно параметров прогноза
ResultsWindow - окно отображения результатов
AgriculturalPredictorApp - основной класс приложения

Поток данных
Загрузка исходных данных из различных источников
Предварительная обработка и очистка данных
Сопоставление регионов с метеорологическими станциями
Извлечение признаков для моделей машинного обучения
Обучение выбранных алгоритмов
Прогнозирование урожайности для выбранного региона
Визуализация и экспорт результатов

Методология прогнозирования
Признаки для моделей
Система использует более 30 различных признаков, включая:

Метеорологические:
Средние, максимальные и минимальные температуры
Амплитуды температур и стандартные отклонения
Сезонные показатели (весна, лето, осень)
Показатели вегетационного периода

Гидрометеорологические:
Среднее давление и его изменчивость
Влажность воздуха и ее сезонные колебания

Экономические:
Средние курсы валют и волатильность
Цены на нефть и их изменчивость
Значения и тренды индекса Московской биржи

Исторические:
Лаговые значения урожайности (до 5 лет)
Тренды урожайности и стандартные отклонения

Метрики оценки моделей
Система оценивает качество моделей по следующим метрикам:
Коэффициент детерминации (R²)
Средняя абсолютная ошибка (MAE)
Доверительные интервалы прогнозов
Вероятность правильности прогноза

Решение проблем
Распространенные проблемы и решения
Ошибки загрузки данных

Проверьте наличие файлов в каталоге agricultural_data/

Убедитесь в правильности форматов Excel файлов

Проверьте соответствие наименований листов

Проблемы с графическим интерфейсом

Убедитесь в установке корректной версии Tkinter

Обновите библиотеки визуализации

Проверьте разрешение экрана и масштабирование

Низкая производительность

Уменьшите количество регионов для обучения моделей

Упростите параметры алгоритмов машинного обучения

Увеличьте объем оперативной памяти

Ошибки картографии

Проверьте наличие интернет-соединения для загрузки карт

Убедитесь в правильности формата файла границ регионов

Проверьте доступность серверов OpenStreetMap

Развитие системы
Планируемые улучшения
Добавление поддержки дополнительных культур

Интеграция с реальными метеорологическими API

Реализация облачных вычислений для сложных моделей

Мобильная версия приложения

Интеграция с системами точного земледелия

Возможности для расширения
Добавление новых источников данных

Реализация дополнительных алгоритмов машинного обучения

Улучшение пользовательского интерфейса

Добавление модуля планирования посевных кампаний

Лицензия
Данный программный продукт предназначен для образовательных и исследовательских целей. Для коммерческого использования требуется согласование с авторами.

Контакты
По вопросам использования, развития системы и сотрудничества обращайтесь к разработчикам.

Agricultural Yield Prediction System
Project Overview
Advanced agricultural crop yield prediction system utilizing machine learning and neural network methods. The system analyzes multi-year yield data, weather conditions, economic indicators, and market indices to generate accurate forecasts.

Key Features
1. Interactive Mapping System
Visualization of Russian Federation regions with borders

Direct region selection on the map

Marker display and region highlighting

OpenStreetMap integration

2. Comprehensive Data Loading and Processing
Regional yield data by year

Meteorological data (temperature, pressure, humidity)

Financial indicators (USD/RUB exchange rates)

Market data (oil prices)

Moscow Exchange stock index

Geodata (region borders in JSON format)

3. Machine Learning Models
Random Forest (decision tree ensemble)

XGBoost (gradient boosting)

Gradient Boosting

LSTM (long short-term memory for time series)

Ensemble forecasting methods

4. Forecasting Functionality
Support for various agricultural crops

Flexible forecast period configuration (1 month to 5 years)

Consideration of multiple influencing factors

Calculation of statistical indicators and confidence intervals

Prediction accuracy probability assessment

5. Results Visualization and Analysis
Yield dynamics charts

Comparative analysis with previous periods

Factor importance analysis

Forecast probability distributions

Analytical report generation

6. Data Export
Complete report saving in Excel format

Graphical material export in PNG format

Detailed analytical reports

Technology Stack
Interface: Tkinter for graphical user interface

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost, LightGBM

Neural Networks: TensorFlow/Keras

Cartography: TkinterMapView

Data Formats: Excel, JSON

Project Structure
text
agricultural_predictor/
│
├── agricultural_data/          # Source data directory
│   ├── Урожайность.xlsx       # Yield data
│   ├── Доллар пример.xlsx     # Currency exchange data
│   ├── oil_prices.xlsx        # Oil price data
│   ├── Индекс МосБиржи.xlsx   # Stock index data
│   ├── weather_data/          # Meteorological data
│   └── regions_borders.json   # Region border geodata
│
├── main.py                    # Main executable file
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation
Installation and Launch
Prerequisites
Python 3.8 or higher

8 GB RAM (recommended)

2 GB free disk space

Dependency Installation
bash
pip install -r requirements.txt
Core Dependencies
txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
tensorflow>=2.11.0
tkintermapview>=1.17
openpyxl>=3.0.0
Application Launch
bash
python main.py
Data Formats
Yield Data
Format: Excel (.xlsx)

Sheet: 'Лист1'

Structure:

First column: region names

Subsequent columns: years (as headers)

Cells: yield values in centners per hectare

Meteorological Data
Format: Excel (.xlsx)

Location: weather_data/ directory

Required fields: temperature, pressure, humidity, date

Supported column names:

Temperature: 'temperature', 'temp'

Pressure: 'pressure', 'davlenie'

Humidity: 'humidity', 'vlaga'

Date: 'date', 'data', 'время', 'дата'

Economic Data
Currency rates: date and exchange rate value

Oil prices: date and price

Moscow Exchange index: date and index value

Geodata
Format: JSON

Structure: polygon coordinates of region borders

Coordinate system: geographic coordinates (latitude, longitude)

User Guide
1. System Initialization
Launch the application

Wait for automatic data loading

If necessary, load region borders manually

2. Model Preparation
Click "Train Models" to initialize machine learning algorithms

The system automatically selects optimal regions for training

Monitor the training process in the information panel

3. Map Interaction
Use the Russian Federation map to select regions

Click on the region of interest to highlight it

Use map zooming and panning tools

4. Forecast Parameter Configuration
Select agricultural crop from the available list

Set the forecast period

Choose the prediction model

Configure additional parameters:

Weather data consideration

Economic indicators consideration

Historical data usage

Historical period length

5. Results Acquisition and Analysis
Review the predicted yield value

Examine the confidence interval and accuracy probability

Analyze yield dynamics on charts

Study factor importance

Utilize generated recommendations

6. Results Export
Save complete report in Excel format

Export charts as images

Use analytical reports for further analysis

System Architecture
System Modules
Config - system configuration parameters

AgriculturalDataLoader - data loader and processor

AdvancedYieldPredictor - machine learning and forecasting module

MapHandler - cartographic operations handler

PredictionDialog - forecast parameters dialog

ResultsWindow - results display window

AgriculturalPredictorApp - main application class

Data Flow
Loading source data from various sources

Preliminary data processing and cleaning

Region matching with meteorological stations

Feature extraction for machine learning models

Training selected algorithms

Yield prediction for selected region

Results visualization and export

Forecasting Methodology
Model Features
The system uses over 30 different features, including:

Meteorological:

Average, maximum, and minimum temperatures

Temperature amplitudes and standard deviations

Seasonal indicators (spring, summer, autumn)

Vegetation period indicators

Hydrometeorological:

Average pressure and its variability

Air humidity and its seasonal fluctuations

Economic:

Average exchange rates and volatility

Oil prices and their variability

Moscow Exchange index values and trends

Historical:

Lagged yield values (up to 5 years)

Yield trends and standard deviations

Model Evaluation Metrics
The system evaluates model quality using the following metrics:

Coefficient of determination (R²)

Mean absolute error (MAE)

Forecast confidence intervals

Prediction accuracy probability

Troubleshooting
Common Issues and Solutions
Data Loading Errors

Verify file presence in the agricultural_data/ directory

Ensure correct Excel file formats

Check sheet name compliance

Graphical Interface Issues

Ensure correct Tkinter version installation

Update visualization libraries

Check screen resolution and scaling

Performance Issues

Reduce the number of regions for model training

Simplify machine learning algorithm parameters

Increase RAM capacity

Cartography Errors

Check internet connection for map loading

Verify region border file format correctness

Check OpenStreetMap server availability

System Development
Planned Improvements
Additional crop support

Integration with real-time meteorological APIs

Cloud computing implementation for complex models

Mobile application version

Integration with precision farming systems

Extension Opportunities
Addition of new data sources

Implementation of additional machine learning algorithms

User interface enhancement

Sowing campaign planning module addition

License
This software product is intended for educational and research purposes. Commercial use requires coordination with the authors.

Contacts
For usage inquiries, system development, and collaboration, please contact the developers.
