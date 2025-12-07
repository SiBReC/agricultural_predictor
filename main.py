import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import json
import os
import glob
import warnings
import threading
import time
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

try:
    import tkintermapview
except ImportError:
    print("Установите tkintermapview: pip install tkintermapview")

warnings.filterwarnings('ignore')


class Config:
    DATA_BASE_PATH = "agricultural_data"
    YIELD_DATA_PATH = os.path.join(DATA_BASE_PATH, "Урожайность.xlsx")
    CURRENCY_DATA_PATH = os.path.join(DATA_BASE_PATH, "Доллар пример.xlsx")
    WEATHER_DATA_DIR = os.path.join(DATA_BASE_PATH, "weather_data")
    OIL_PRICE_DATA_PATH = os.path.join(DATA_BASE_PATH, "oil_prices.xlsx")
    MOSBIR_INDEX_DATA_PATH = os.path.join(DATA_BASE_PATH, "Индекс МосБиржи.xlsx")
    REGIONS_BORDERS_PATH = os.path.join(DATA_BASE_PATH, "regions_borders.json")

    MODEL_TYPES = {
        "LSTM": "lstm", "GRU": "gru", "Transformer": "transformer",
        "Ensemble": "ensemble", "XGBoost": "xgboost", "RandomForest": "random_forest"
    }

    FORECAST_PERIODS = {
        "1 месяц": 1, "6 месяцев": 6, "1 год": 12,
        "2 года": 24, "3 года": 36, "5 лет": 60
    }

    CROPS = {
        "Пшеница": "wheat", "Ячмень": "barley", "Рожь": "rye",
        "Овес": "oats", "Кукуруза": "corn", "Подсолнечник": "sunflower",
        "Соя": "soy", "Рапс": "rapeseed"
    }

    COLORS = {
        "primary": "#2c3e50", "secondary": "#34495e", "accent": "#3498db",
        "success": "#27ae60", "warning": "#f39c12", "danger": "#e74c3c",
        "light": "#ecf0f1", "dark": "#2c3e50"
    }


class AgriculturalDataLoader:
    def __init__(self):
        self.yield_data = None
        self.weather_data = {}
        self.currency_data = {}
        self.oil_data = None
        self.mosbir_index_data = None
        self.region_coordinates = {}
        self.loaded = False
        self.region_weather_mapping = {}
        self.region_folders = {}

    def load_all_data(self):
        print("🔄 Начинаю загрузку данных...")
        try:
            os.makedirs(Config.DATA_BASE_PATH, exist_ok=True)
            os.makedirs(Config.WEATHER_DATA_DIR, exist_ok=True)

            self.load_yield_data()
            self.load_weather_data()
            self.load_currency_data()
            self.load_oil_price_data()
            self.load_mosbir_index_data()
            self.create_region_weather_mapping()

            self.loaded = True
            print("✅ Все данные успешно загружены!")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            self.loaded = False
            return False

    def load_yield_data(self):
        try:
            if os.path.exists(Config.YIELD_DATA_PATH):
                self.yield_data = pd.read_excel(Config.YIELD_DATA_PATH, sheet_name='Лист1', header=0)
                yield_data_long = []

                for idx, row in self.yield_data.iterrows():
                    region_name = str(row.iloc[0]).strip()
                    for col in self.yield_data.columns[1:]:
                        if str(col).isdigit():
                            year = int(col)
                            yield_value = row[col]
                            if pd.notna(yield_value) and yield_value != '':
                                yield_data_long.append({
                                    'region': region_name,
                                    'year': year,
                                    'yield': float(yield_value)
                                })

                self.yield_data = pd.DataFrame(yield_data_long)
                print(f"✅ Данные урожайности загружены: {len(self.yield_data)} записей")
                print(f"📊 Доступные регионы: {self.get_available_regions()}")
            else:
                raise FileNotFoundError(f"Файл урожайности не найден: {Config.YIELD_DATA_PATH}")
        except Exception as e:
            print(f"❌ Ошибка загрузки урожайности: {e}")
            raise

    def load_weather_data(self):
        try:
            weather_files = []
            for root, dirs, files in os.walk(Config.WEATHER_DATA_DIR):
                for file in files:
                    if file.endswith(('.xlsx', '.xls')):
                        weather_files.append(os.path.join(root, file))

            if not weather_files:
                print(f"⚠️ Файлы погоды не найдены в {Config.WEATHER_DATA_DIR}")
                return

            loaded_files = 0
            for file_path in weather_files:
                try:
                    filename = os.path.basename(file_path)
                    city_name = self.extract_city_name(filename)
                    if not city_name:
                        print(f"⚠️ Не удалось извлечь название города из {filename}")
                        continue

                    folder_name = os.path.basename(os.path.dirname(file_path))
                    if folder_name == Config.WEATHER_DATA_DIR:
                        folder_name = "Общие"

                    df = self.load_single_weather_file(file_path)
                    if df is not None and not df.empty:
                        key = f"{folder_name}_{city_name}"
                        self.weather_data[key] = df
                        self.region_folders[key] = folder_name
                        loaded_files += 1
                        print(f"✅ Погода {folder_name}/{city_name}: {len(df)} записей")
                    else:
                        print(f"⚠️ Не удалось загрузить данные из {filename}")
                except Exception as e:
                    print(f"❌ Ошибка загрузки погоды {file_path}: {e}")

            print(f"✅ Загружено погодных данных: {loaded_files} файлов")
            if loaded_files == 0:
                print("⚠️ Не удалось загрузить ни одного файла погоды, продолжаем без погодных данных")
        except Exception as e:
            print(f"❌ Общая ошибка загрузки погоды: {e}")

    def extract_city_name(self, filename):
        import re
        name = os.path.splitext(filename)[0]
        name = re.sub(r'\d{4}.*\d{4}', '', name)
        name = re.sub(r'\d+', '', name)
        name = name.replace('_', ' ').replace('-', ' ').strip()
        name = re.sub(r'\s+', ' ', name)
        return name if name else "Неизвестный"

    def load_single_weather_file(self, file_path):
        try:
            print(f"📖 Чтение файла: {os.path.basename(file_path)}")

            sheets_to_try = ['Архив Погоды rp5', 'Лист1', 'Sheet1', 0]
            df = None

            for sheet in sheets_to_try:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    if df is not None and not df.empty:
                        print(f"✅ Успешно прочитан лист: {sheet}")
                        break
                except Exception as e:
                    print(f"❌ Ошибка чтения листа {sheet}: {e}")
                    continue

            if df is None or df.empty:
                print(f"⚠️ Не удалось прочитать файл {file_path}")
                return None

            print(f"📊 Структура данных ({os.path.basename(file_path)}):")
            print(f"   Колонки: {list(df.columns)}")
            print(f"   Всего строк: {len(df)}")
            if len(df) > 0:
                print(f"   Первые строки:")
                print(df.head(2))

            df.columns = [str(col).lower().strip() for col in df.columns]
            print(f"📝 Нормализованные колонки: {list(df.columns)}")

            df = self.process_dates(df)

            if df.empty:
                print(f"⚠️ После обработки дат не осталось записей в файле {file_path}")
                return None

            print("🔄 Переименование колонок...")

            rename_dict = {}

            column_mapping = {
                'temp': 'temperature',
                'davlenie': 'pressure',
                'vlaga': 'humidity',
                'date': 'date'
            }

            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    rename_dict[old_name] = new_name
                    print(f"   {old_name} -> {new_name}")

            if rename_dict:
                df = df.rename(columns=rename_dict)

            print(f"📝 Колонки после переименования: {list(df.columns)}")

            available_columns = [col for col in ['temperature', 'pressure', 'humidity', 'date'] if col in df.columns]
            if not available_columns:
                print(f"⚠️ Не найдены нужные колонки в файле {file_path}")
                return None

            df = df[available_columns]

            for col in ['temperature', 'pressure', 'humidity']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if col == 'temperature':
                        df[col] = df[col].apply(lambda x: x if -50 <= x <= 50 else np.nan)
                    elif col == 'pressure':
                        df[col] = df[col].apply(lambda x: x if 700 <= x <= 800 else np.nan)
                    elif col == 'humidity':
                        df[col] = df[col].apply(lambda x: x if 0 <= x <= 100 else np.nan)

            numeric_cols = [col for col in ['temperature', 'pressure', 'humidity'] if col in df.columns]
            if numeric_cols:
                initial_count = len(df)
                df = df.dropna(subset=numeric_cols, how='all')
                final_count = len(df)
                print(f"📊 Очистка данных: {initial_count} -> {final_count} записей")

            print(f"✅ Успешно обработан файл {os.path.basename(file_path)}: {len(df)} записей")
            return df

        except Exception as e:
            print(f"❌ Критическая ошибка обработки файл {file_path}: {e}")
            import traceback
            print(f"🔍 Детали ошибки: {traceback.format_exc()}")
            return None

    def process_dates(self, df):
        date_columns = [col for col in df.columns if any(x in col.lower() for x in ['date', 'data', 'время', 'дата'])]

        if not date_columns:
            for col in df.columns:
                if len(df) > 0:
                    sample_val = str(df[col].iloc[0])
                    if any(x in sample_val for x in ['202', '201', '200', '/', '-', ':']):
                        date_columns = [col]
                        break

        if not date_columns:
            print("⚠️ Не найдена колонку с датами")
            return df

        date_col = date_columns[0]
        print(f"📅 Использую колонку '{date_col}' для дат")

        df[date_col] = df[date_col].astype(str).str.strip()

        sample_dates = df[date_col].head(3).tolist()
        print(f"📅 Примеры дат: {sample_dates}")

        def parse_custom_date(date_str):
            try:
                if ' ' in date_str:
                    date_part, time_part = date_str.split(' ', 1)
                    day, month, year = date_part.split('.')
                    hour, minute = time_part.split(':')
                    return pd.Timestamp(year=int(year), month=int(month), day=int(day),
                                        hour=int(hour), minute=int(minute))
                else:
                    day, month, year = date_str.split('.')
                    return pd.Timestamp(year=int(year), month=int(month), day=int(day))
            except Exception as e:
                print(f"❌ Ошибка парсинга даты '{date_str}': {e}")
                return pd.NaT

        print("🔄 Применяю кастомный парсер дат...")
        df['date'] = df[date_col].apply(parse_custom_date)

        valid_dates = df['date'].notna().sum()
        print(f"✅ Валидных дат распознано: {valid_dates} из {len(df)}")

        if valid_dates == 0:
            print("⚠️ Кастомный парсер не сработал, пробую стандартные методы...")

            date_formats = [
                '%d.%m.%Y %H:%M', '%d.%m.%Y %H:%M:%S', '%d.%m.%Y',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d',
                '%d/%m/%Y %H:%M', '%d/%m/%Y',
                '%m/%d/%Y %H:%M', '%m/%d/%Y',
                '%d.%m.%Y %H.%M'
            ]

            for fmt in date_formats:
                try:
                    temp_dates = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                    valid_count = temp_dates.notna().sum()
                    print(f"  Формат {fmt}: {valid_count} валидных дат")

                    if valid_count > 0:
                        df['date'] = temp_dates
                        break
                except Exception as e:
                    print(f"  Ошибка формата {fmt}: {e}")
                    continue

            if df['date'].isna().all():
                print("⚠️ Не удалось распарсить даты форматами, пробую автоматическое определение...")
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')

        initial_count = len(df)
        df = df.dropna(subset=['date'])
        final_count = len(df)
        print(f"📊 Даты обработаны: {initial_count} -> {final_count} записей")

        return df

    def load_currency_data(self):
        try:
            if os.path.exists(Config.CURRENCY_DATA_PATH):
                sheets_to_try = ['RC', 'Лист1', 'Sheet1', 0]
                df = None

                for sheet in sheets_to_try:
                    try:
                        df = pd.read_excel(Config.CURRENCY_DATA_PATH, sheet_name=sheet)
                        if df is not None and not df.empty:
                            break
                    except:
                        continue

                if df is None:
                    raise Exception("Не удалось прочитать файл курсов")

                df.columns = [str(col).lower().strip() for col in df.columns]

                date_columns = [col for col in df.columns if any(x in col for x in ['date', 'data', 'дата'])]
                if date_columns:
                    df['date'] = pd.to_datetime(df[date_columns[0]], errors='coerce')

                rate_columns = [col for col in df.columns if any(x in col for x in ['curs', 'курс', 'rate', 'usd'])]
                if rate_columns:
                    df['usd_rub'] = pd.to_numeric(df[rate_columns[0]], errors='coerce')

                df = df.dropna(subset=['date', 'usd_rub'])
                self.currency_data['USD_RUB'] = df
                print(f"✅ Курсы валют загружены: {len(df)} записей")
            else:
                print(f"⚠️ Файл курсов не найден: {Config.CURRENCY_DATA_PATH}")
        except Exception as e:
            print(f"❌ Ошибка загрузки курсов: {e}")

    def load_oil_price_data(self):
        try:
            if os.path.exists(Config.OIL_PRICE_DATA_PATH):
                self.oil_data = pd.read_excel(Config.OIL_PRICE_DATA_PATH)
                print(f"✅ Данные по нефти загружены: {len(self.oil_data)} записей")
            else:
                print("⚠️ Файл нефти не найден, продолжаем без данных по нефти")
                self.oil_data = pd.DataFrame()
        except Exception as e:
            print(f"❌ Ошибка загрузки нефти: {e}")
            self.oil_data = pd.DataFrame()

    def load_mosbir_index_data(self):
        try:
            if os.path.exists(Config.MOSBIR_INDEX_DATA_PATH):
                sheets_to_try = ['Прошлые данные - Индекс МосБирж', 'Лист1', 'Sheet1', 0]
                df = None

                for sheet in sheets_to_try:
                    try:
                        df = pd.read_excel(Config.MOSBIR_INDEX_DATA_PATH, sheet_name=sheet)
                        if df is not None and not df.empty:
                            print(f"✅ Успешно прочитан лист: {sheet}")
                            break
                    except Exception as e:
                        print(f"❌ Ошибка чтения листа {sheet}: {e}")
                        continue

                if df is None:
                    raise Exception("Не удалось прочитать файл индекса МосБиржи")

                df.columns = [str(col).lower().strip() for col in df.columns]
                print(f"📊 Колонки индекса МосБиржи: {list(df.columns)}")
                print(f"📊 Первые строки данных:")
                print(df.head(3))

                date_columns = [col for col in df.columns if any(x in col for x in ['date', 'data', 'дата', 'время'])]
                if date_columns:
                    print(f"📅 Найдены колонки с датами: {date_columns}")
                    for date_col in date_columns:
                        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                        valid_dates = df['date'].notna().sum()
                        print(f"  Колонка '{date_col}': {valid_dates} валидных дат")
                        if valid_dates > 0:
                            break
                else:
                    print("⚠️ Явная колонка с датой не найдена, пробую первую колонку")
                    df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

                price_columns = [col for col in df.columns if
                                 any(x in col for x in ['цена', 'price', 'close', 'индекс', 'index', 'значение'])]
                if price_columns:
                    print(f"💰 Найдены колонки с ценами: {price_columns}")
                    price_col = price_columns[0]
                    print(f"💰 Использую колонку: {price_col}")

                    sample_values = df[price_col].head(3).tolist()
                    print(f"💰 Примеры значений: {sample_values}")

                    price_series = df[price_col].astype(str)
                    price_series = price_series.str.replace(' ', '', regex=False)
                    price_series = price_series.str.replace(',', '.', regex=False)

                    def clean_number(x):
                        try:
                            parts = x.split('.')
                            if len(parts) > 2:
                                whole_part = ''.join(parts[:-1])
                                decimal_part = parts[-1]
                                return f"{whole_part}.{decimal_part}"
                            return x
                        except:
                            return x

                    price_series = price_series.apply(clean_number)
                    df['mosbir_index'] = pd.to_numeric(price_series, errors='coerce')

                    print(f"💰 После очистки примеры: {df['mosbir_index'].head(3).tolist()}")
                else:
                    print("⚠️ Явная колонка с ценой не найдена, пробую вторую колонку")
                    price_series = df.iloc[:, 1].astype(str)
                    price_series = price_series.str.replace(' ', '', regex=False)
                    price_series = price_series.str.replace(',', '.', regex=False)
                    df['mosbir_index'] = pd.to_numeric(price_series, errors='coerce')

                initial_count = len(df)
                df = df.dropna(subset=['date', 'mosbir_index'])
                final_count = len(df)

                self.mosbir_index_data = df
                print(f"✅ Индекс МосБиржи загружен: {final_count} записей (было {initial_count})")

                if not df.empty:
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    print(
                        f"📅 Диапазон дат индекса МосБиржи: {min_date.strftime('%d.%m.%Y')} - {max_date.strftime('%d.%m.%Y')}")
                    print(f"📈 Диапазон значений: {df['mosbir_index'].min():.2f} - {df['mosbir_index'].max():.2f}")

                    df['year'] = df['date'].dt.year
                    yearly_stats = df.groupby('year')['mosbir_index'].agg(['count', 'min', 'max', 'mean']).round(2)
                    print(f"📊 Статистика по годам:\n{yearly_stats}")
                else:
                    print("⚠️ Нет данных индекса МосБиржи после очистки")

                    print("🔍 Диагностика проблем:")
                    print(f"   Всего строк: {initial_count}")
                    print(f"   Пустых дат: {df['date'].isna().sum()}")
                    print(f"   Пустых значений индекса: {df['mosbir_index'].isna().sum()}")
            else:
                print(f"⚠️ Файл индекса МосБиржи не найден: {Config.MOSBIR_INDEX_DATA_PATH}")
                self.mosbir_index_data = pd.DataFrame()
        except Exception as e:
            print(f"❌ Ошибка загрузки индекса МосБиржи: {e}")
            import traceback
            print(f"🔍 Детали ошибки: {traceback.format_exc()}")
            self.mosbir_index_data = pd.DataFrame()

    def create_region_weather_mapping(self):
        available_regions = [str(region).strip() for region in self.yield_data['region'].unique()]

        print("🔗 Создание сопоставления регионов и погодных данных...")

        for region in available_regions:
            region_lower = region.lower().strip()
            matching_weather_keys = []

            for weather_key, folder_name in self.region_folders.items():
                folder_lower = folder_name.lower()

                if (region_lower in folder_lower or
                        folder_lower in region_lower or
                        self.regions_similar(region_lower, folder_lower)):
                    matching_weather_keys.append(weather_key)

            if not matching_weather_keys:
                for weather_key in self.weather_data.keys():
                    city_part = weather_key.split('_')[-1].lower()
                    region_clean = self.clean_region_name(region_lower)

                    if (city_part in region_clean or
                            region_clean in city_part or
                            self.regions_similar(region_clean, city_part)):
                        matching_weather_keys.append(weather_key)

            if matching_weather_keys:
                selected_weather = matching_weather_keys[0]
                self.region_weather_mapping[region] = selected_weather
                print(f"✅ Сопоставление: {region} -> {selected_weather}")
            else:
                for weather_key in self.weather_data.keys():
                    if any(word in region_lower for word in weather_key.lower().split('_')):
                        matching_weather_keys.append(weather_key)
                        break

                if matching_weather_keys:
                    selected_weather = matching_weather_keys[0]
                    self.region_weather_mapping[region] = selected_weather
                    print(f"🔗 Приблизительное сопоставление: {region} -> {selected_weather}")
                else:
                    print(f"❌ Нет погодных данных для региона: {region}")

        print(f"✅ Создан mapping регионов и погодных станций: {len(self.region_weather_mapping)} регионов")

    def clean_region_name(self, region_name):
        common_words = ['область', 'край', 'республика', 'автономный', 'округ', 'город']
        words = region_name.split()
        cleaned = [word for word in words if word not in common_words]
        return ' '.join(cleaned)

    def regions_similar(self, region1, region2):
        r1_clean = self.clean_region_name(region1)
        r2_clean = self.clean_region_name(region2)

        return (r1_clean in r2_clean or r2_clean in r1_clean or
                r1_clean.replace(' ', '') in r2_clean.replace(' ', '') or
                r2_clean.replace(' ', '') in r1_clean.replace(' ', ''))

    def get_region_data(self, region_name):
        if not self.loaded:
            return None

        normalized_region = str(region_name).strip()
        region_yield = self.yield_data[self.yield_data['region'] == normalized_region]

        if region_yield.empty:
            print(f"❌ Нет данных урожайности для региона: {normalized_region}")
            return None

        weather_key = self.region_weather_mapping.get(normalized_region)
        weather_data = self.weather_data.get(weather_key) if weather_key else None

        return {
            'yield': region_yield,
            'weather': weather_data,
            'currency': self.currency_data.get('USD_RUB'),
            'oil': self.oil_data,
            'mosbir_index': self.mosbir_index_data,
            'weather_city': weather_key
        }

    def get_available_regions(self):
        if self.yield_data is not None:
            return [str(region).strip() for region in self.yield_data['region'].unique().tolist()]
        return []


class AdvancedYieldPredictor:
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        self.feature_names = []
        self.expected_features = []
        self.lookback_period = 5
        self.all_possible_features = self.get_all_possible_features()

    def get_all_possible_features(self):
        features = [
            'avg_temperature', 'max_temperature', 'min_temperature',
            'temp_amplitude', 'temp_std',
            'spring_avg_temp', 'spring_max_temp',
            'summer_avg_temp', 'summer_max_temp',
            'autumn_avg_temp', 'growing_season_avg_temp',
            'avg_pressure', 'pressure_std', 'min_pressure', 'max_pressure',
            'avg_humidity', 'humidity_std', 'min_humidity', 'max_humidity',
            'spring_avg_humidity', 'summer_avg_humidity',
            'avg_usd_rate', 'usd_volatility', 'min_usd_rate', 'max_usd_rate',
            'avg_oil_price', 'oil_price_volatility',
            'avg_mosbir_index', 'mosbir_volatility', 'min_mosbir_index', 'max_mosbir_index',
            'mosbir_trend', 'mosbir_annual_return',
            'yield_trend', 'yield_std'
        ]

        for i in range(1, self.lookback_period + 1):
            features.append(f'yield_lag_{i}')

        return features

    def prepare_features(self, region_data, lookback_period=None):
        if lookback_period is None:
            lookback_period = self.lookback_period

        if region_data is None:
            return None, None
        yield_data = region_data['yield']
        weather_data = region_data['weather']
        currency_data = region_data['currency']
        oil_data = region_data['oil']
        mosbir_index_data = region_data['mosbir_index']

        if yield_data.empty:
            return None, None

        combined_data = []
        for _, row in yield_data.iterrows():
            year = row['year']
            yield_value = row['yield']
            features = self.extract_features_for_year(year, weather_data, currency_data, oil_data, mosbir_index_data,
                                                      yield_data, lookback_period)
            if features:
                features['target'] = yield_value
                combined_data.append(features)

        if not combined_data:
            return None, None

        feature_df = pd.DataFrame(combined_data)

        feature_df = feature_df.fillna(0)

        for feature in self.all_possible_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0.0

        if hasattr(self, 'expected_features') and self.expected_features:
            expected_cols = [col for col in self.expected_features if col in feature_df.columns]
            if 'target' in feature_df.columns:
                expected_cols.append('target')
            feature_df = feature_df[expected_cols]
        else:
            self.expected_features = [col for col in self.all_possible_features if col in feature_df.columns]

        self.feature_names = [col for col in feature_df.columns if col != 'target']
        X = feature_df[self.feature_names]
        y = feature_df['target'] if 'target' in feature_df.columns else None

        return X, y

    def extract_features_for_year(self, year, weather, currency, oil, mosbir_index, yield_history, lookback):
        features = {}
        try:
            for feature in self.all_possible_features:
                features[feature] = 0.0

            if weather is not None and not weather.empty:
                year_weather = weather[weather['date'].dt.year == year]
                if not year_weather.empty:
                    spring = year_weather[year_weather['date'].dt.month.isin([3, 4, 5])]
                    summer = year_weather[year_weather['date'].dt.month.isin([6, 7, 8])]
                    autumn = year_weather[year_weather['date'].dt.month.isin([9, 10, 11])]

                    if 'temperature' in year_weather.columns:
                        features['avg_temperature'] = year_weather['temperature'].mean()
                        features['max_temperature'] = year_weather['temperature'].max()
                        features['min_temperature'] = year_weather['temperature'].min()
                        features['temp_amplitude'] = features['max_temperature'] - features['min_temperature']
                        features['temp_std'] = year_weather['temperature'].std()

                        if not spring.empty:
                            features['spring_avg_temp'] = spring['temperature'].mean()
                            features['spring_max_temp'] = spring['temperature'].max()
                        if not summer.empty:
                            features['summer_avg_temp'] = summer['temperature'].mean()
                            features['summer_max_temp'] = summer['temperature'].max()
                        if not autumn.empty:
                            features['autumn_avg_temp'] = autumn['temperature'].mean()

                        growing_season = year_weather[year_weather['date'].dt.month.isin([4, 5, 6, 7, 8, 9])]
                        if not growing_season.empty:
                            features['growing_season_avg_temp'] = growing_season['temperature'].mean()

                    if 'pressure' in year_weather.columns:
                        features['avg_pressure'] = year_weather['pressure'].mean()
                        features['pressure_std'] = year_weather['pressure'].std()
                        features['min_pressure'] = year_weather['pressure'].min()
                        features['max_pressure'] = year_weather['pressure'].max()

                    if 'humidity' in year_weather.columns:
                        features['avg_humidity'] = year_weather['humidity'].mean()
                        features['humidity_std'] = year_weather['humidity'].std()
                        features['min_humidity'] = year_weather['humidity'].min()
                        features['max_humidity'] = year_weather['humidity'].max()

                        if not summer.empty:
                            features['summer_avg_humidity'] = summer['humidity'].mean()
                        if not spring.empty:
                            features['spring_avg_humidity'] = spring['humidity'].mean()

            if currency is not None and not currency.empty:
                year_currency = currency[currency['date'].dt.year == year]
                if not year_currency.empty and 'usd_rub' in year_currency.columns:
                    features['avg_usd_rate'] = year_currency['usd_rub'].mean()
                    features['usd_volatility'] = year_currency['usd_rub'].std()
                    features['min_usd_rate'] = year_currency['usd_rub'].min()
                    features['max_usd_rate'] = year_currency['usd_rub'].max()

            if mosbir_index is not None and not mosbir_index.empty:
                year_mosbir = mosbir_index[mosbir_index['date'].dt.year == year]
                if not year_mosbir.empty and 'mosbir_index' in year_mosbir.columns:
                    features['avg_mosbir_index'] = year_mosbir['mosbir_index'].mean()
                    features['mosbir_volatility'] = year_mosbir['mosbir_index'].std()
                    features['min_mosbir_index'] = year_mosbir['mosbir_index'].min()
                    features['max_mosbir_index'] = year_mosbir['mosbir_index'].max()

                    if len(year_mosbir) > 1:
                        dates_numeric = (year_mosbir['date'] - year_mosbir['date'].min()).dt.days
                        if dates_numeric.std() > 0:
                            trend_coeff = np.polyfit(dates_numeric, year_mosbir['mosbir_index'], 1)[0]
                            features['mosbir_trend'] = trend_coeff

                    if len(year_mosbir) >= 2:
                        first_value = year_mosbir.sort_values('date')['mosbir_index'].iloc[0]
                        last_value = year_mosbir.sort_values('date')['mosbir_index'].iloc[-1]
                        if first_value > 0:
                            features['mosbir_annual_return'] = (last_value - first_value) / first_value * 100

            for i in range(1, self.lookback_period + 1):
                prev_year = year - i
                prev_yield_data = yield_history[yield_history['year'] == prev_year]
                if not prev_yield_data.empty:
                    features[f'yield_lag_{i}'] = prev_yield_data['yield'].iloc[0]

            recent_years = [year - i for i in range(1, min(6, self.lookback_period + 1))]
            recent_yields = []
            for y in recent_years:
                yield_val = yield_history[yield_history['year'] == y]['yield']
                if not yield_val.empty:
                    recent_yields.append(yield_val.iloc[0])

            if len(recent_yields) > 1:
                features['yield_trend'] = np.polyfit(range(len(recent_yields)), recent_yields, 1)[0]
                features['yield_std'] = np.std(recent_yields)

            if oil is not None and not oil.empty:
                year_oil = oil[oil['date'].dt.year == year]
                if not year_oil.empty and 'oil_price' in year_oil.columns:
                    features['avg_oil_price'] = year_oil['oil_price'].mean()
                    features['oil_price_volatility'] = year_oil['oil_price'].std()

            for key in features:
                if pd.isna(features[key]):
                    features[key] = 0

            return features
        except Exception as e:
            print(f"Ошибка извлечения признаков для {year}: {e}")
            import traceback
            print(f"🔍 Детали ошибки: {traceback.format_exc()}")
            return features

    def train_models(self, X, y):
        if X is None or y is None or len(X) < 8:
            print("Недостаточно данных для обучения")
            return False

        try:
            self.expected_features = X.columns.tolist()

            non_constant_features = X.columns[X.std() > 0].tolist()
            if len(non_constant_features) < len(X.columns):
                print(f"Удалены константные признаки: {set(X.columns) - set(non_constant_features)}")
                X = X[non_constant_features]

            self.feature_names = X.columns.tolist()
            self.expected_features = self.feature_names

            self.scalers['feature'] = StandardScaler()
            X_scaled = self.scalers['feature'].fit_transform(X)
            self.scalers['target'] = StandardScaler()
            y_scaled = self.scalers['target'].fit_transform(y.values.reshape(-1, 1)).flatten()

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.15, random_state=42, shuffle=False
            )

            model_performance = {}

            models_to_train = {
                'random_forest': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=3, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
            }

            for name, model in models_to_train.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    model_performance[name] = {'r2': score, 'mae': mae}
                    self.models[name] = model
                    print(f"✅ Модель {name} обучена, R²: {score:.3f}, MAE: {mae:.3f}")
                except Exception as e:
                    print(f"Ошибка обучения {name}: {e}")

            self.training_history.append({
                'timestamp': pd.Timestamp.now(),
                'models_trained': list(self.models.keys()),
                'performance': model_performance,
                'best_model': max(model_performance, key=lambda x: model_performance[x]['r2']) if model_performance else None
            })

            self.is_trained = True
            print(f"✅ Модели обучены. Лучшая модель: {self.training_history[-1]['best_model']}")
            return True
        except Exception as e:
            print(f"❌ Ошибка обучения моделей: {e}")
            return False

    def predict(self, X, method='ensemble'):
        if not self.is_trained or X is None or len(self.models) == 0:
            return None, 0.0, 0.0

        try:
            if hasattr(X, 'columns'):
                X_aligned = pd.DataFrame(columns=self.feature_names)

                for col in self.feature_names:
                    if col in X.columns:
                        X_aligned[col] = X[col]
                    else:
                        X_aligned[col] = 0.0

                X = X_aligned

            X_scaled = self.scalers['feature'].transform(X)
            predictions = []
            model_weights = {}

            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                    if self.training_history and 'performance' in self.training_history[-1]:
                        perf = self.training_history[-1]['performance'].get(name, {'r2': 0.5})['r2']
                        model_weights[name] = max(0.1, perf)
                    else:
                        model_weights[name] = 0.5
                except Exception as e:
                    print(f"Ошибка предсказания моделью {name}: {e}")
                    continue

            if not predictions:
                return None, 0.0, 0.0

            if method == 'ensemble' and len(predictions) > 1:
                total_weight = sum(model_weights.values())
                ensemble_pred = np.zeros_like(predictions[0])
                for i, pred in enumerate(predictions):
                    weight = list(model_weights.values())[i] / total_weight
                    ensemble_pred += pred * weight
                final_prediction_scaled = ensemble_pred
            else:
                final_prediction_scaled = predictions[0]

            final_prediction = self.scalers['target'].inverse_transform(
                final_prediction_scaled.reshape(-1, 1)
            ).flatten()[0]

            confidence = min(0.85, max(0.5, sum(model_weights.values()) / len(model_weights)))
            deviation = abs(final_prediction * 0.08)

            return final_prediction, confidence, deviation
        except Exception as e:
            print(f"❌ Ошибка прогнозирования: {e}")
            import traceback
            print(f"🔍 Детали ошибки: {traceback.format_exc()}")
            return None, 0.0, 0.0

    def calculate_feature_importance(self, X, feature_names):
        if 'random_forest' in self.models:
            try:
                rf_model = self.models['random_forest']
                importance = rf_model.feature_importances_
                self.feature_importance = dict(zip(feature_names, importance))
                self.feature_importance = dict(
                    sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
                return self.feature_importance
            except:
                pass
        return {}

    def get_model_performance(self):
        if self.training_history:
            return self.training_history[-1]['performance']
        return {}


class MapHandler:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.map_widget = None
        self.regions_data = {}
        self.loaded_regions = {}
        self.active_polygons = []
        self.active_markers = []
        self.current_marker = None
        self.current_highlighted_region = None
        self.setup_map()

    def setup_map(self):
        self.map_widget = tkintermapview.TkinterMapView(self.parent_frame, width=1200, height=600)
        self.map_widget.pack(fill=tk.BOTH, expand=True)
        self.map_widget.set_position(55.7558, 37.6173)
        self.map_widget.set_zoom(4)
        self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")

    def load_regions_data(self, filename=None):
        if filename is None:
            filename = Config.REGIONS_BORDERS_PATH

        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.regions_data = json.load(f)

                self.loaded_regions = {}
                for region_name, region_data in self.regions_data.items():
                    if isinstance(region_data, dict):
                        if "0" in region_data:
                            coords = region_data["0"]
                            if coords and len(coords) > 0 and isinstance(coords[0], list) and len(coords[0]) == 2:
                                self.loaded_regions[region_name] = coords
                        elif "coordinates" in region_data:
                            coords = region_data["coordinates"]
                            if isinstance(coords, list) and len(coords) > 0:
                                if isinstance(coords[0][0], list):
                                    self.loaded_regions[region_name] = coords[0]
                                else:
                                    self.loaded_regions[region_name] = coords
                    elif isinstance(region_data, list):
                        if len(region_data) > 0 and isinstance(region_data[0], list):
                            self.loaded_regions[region_name] = region_data

                print(f"✅ Границы регионов загружены: {len(self.loaded_regions)} регионов")
                return len(self.loaded_regions)

            except Exception as e:
                print(f"❌ Ошибка загрузки границ регионов: {e}")
                self.create_basic_regions()
                return len(self.loaded_regions)
        else:
            print("⚠️ Файл границ не найден, создаю базовые регионы...")
            self.create_basic_regions()
            return len(self.loaded_regions)

    def create_basic_regions(self):
        basic_regions = {
            "Ростовская область": [
                [47.222, 39.718], [48.0, 40.0], [48.5, 41.0], [47.5, 42.0],
                [46.0, 41.5], [45.5, 40.0], [46.0, 39.0], [47.222, 39.718]
            ],
            "Краснодарский край": [
                [45.035, 38.975], [46.0, 39.5], [47.0, 39.0], [47.5, 38.0],
                [46.5, 37.0], [45.0, 37.5], [44.5, 38.0], [45.035, 38.975]
            ],
            "Ставропольский край": [
                [45.043, 41.969], [46.0, 42.5], [47.0, 42.0], [47.5, 41.0],
                [46.5, 40.0], [45.5, 40.5], [44.5, 41.0], [45.043, 41.969]
            ]
        }
        self.loaded_regions = basic_regions

    def show_region_borders(self, region_name=None):
        self.clear_map()

        if not self.loaded_regions:
            print("⚠️ Нет загруженных границ регионов")
            return

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

        for i, (name, coordinates) in enumerate(self.loaded_regions.items()):
            if region_name and name != region_name:
                continue

            color = colors[i % len(colors)]

            try:
                polygon = self.map_widget.set_polygon(
                    coordinates,
                    fill_color=color,
                    outline_color=color,
                    border_width=2,
                    name=name
                )
                self.active_polygons.append(polygon)
            except Exception as e:
                print(f"Ошибка отображения региона {name}: {e}")

    def show_all_regions_borders(self):
        self.show_region_borders()

    def highlight_region(self, region_name):
        self.clear_map()

        if region_name in self.loaded_regions:
            coordinates = self.loaded_regions[region_name]
            try:
                polygon = self.map_widget.set_polygon(
                    coordinates,
                    fill_color="#3498db",
                    outline_color="#3498db",
                    border_width=3,
                    name=region_name
                )
                self.active_polygons.append(polygon)
                self.current_highlighted_region = region_name

                if coordinates:
                    avg_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
                    avg_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
                    self.map_widget.set_position(avg_lat, avg_lon)
                    self.map_widget.set_zoom(7)

            except Exception as e:
                print(f"Ошибка подсветки региона {region_name}: {e}")
        else:
            print(f"⚠️ Регион {region_name} не найден в загруженных границах")

    def add_marker(self, lat, lon, text=""):
        if self.current_marker:
            try:
                self.current_marker.delete()
            except:
                pass

        try:
            self.current_marker = self.map_widget.set_marker(lat, lon, text=text)
            self.active_markers.append(self.current_marker)
            return self.current_marker
        except Exception as e:
            print(f"Ошибка добавления маркера: {e}")
            return None

    def clear_map(self):
        for polygon in self.active_polygons:
            try:
                polygon.delete()
            except Exception as e:
                print(f"Ошибка удаления полигона: {e}")
        self.active_polygons.clear()

        for marker in self.active_markers:
            try:
                marker.delete()
            except Exception as e:
                print(f"Ошибка удаления маркера: {e}")
        self.active_markers.clear()

        if self.current_marker:
            try:
                self.current_marker.delete()
            except Exception as e:
                print(f"Ошибка удаления текущего маркера: {e}")
            self.current_marker = None

        self.current_highlighted_region = None

    def find_region_by_coords(self, lat, lon):
        if not self.loaded_regions:
            return self.find_region_by_coords_fallback(lat, lon)

        for region_name, coordinates in self.loaded_regions.items():
            if self.point_in_polygon(lat, lon, coordinates):
                return region_name

        return self.find_region_by_coords_fallback(lat, lon)

    def point_in_polygon(self, lat, lon, polygon):
        if not polygon or len(polygon) < 3:
            return False

        inside = False
        j = len(polygon) - 1

        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > lon) != (yj > lon)) and (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def find_region_by_coords_fallback(self, lat, lon):
        region_centers = {
            "Ростовская область": (47.222, 39.718), "Краснодарский край": (45.035, 38.975),
            "Ставропольский край": (45.043, 41.969), "Воронежская область": (51.672, 39.184),
            "Белгородская область": (50.597, 36.588), "Курская область": (51.730, 36.193),
            "Орловская область": (52.967, 36.069), "Тамбовская область": (52.721, 41.453),
            "Липецкая область": (52.608, 39.599), "Московская область": (55.755, 37.617),
            "Ленинградская область": (59.939, 30.315), "Новосибирская область": (55.030, 82.920),
            "Алтайский край": (53.348, 83.776), "Республика Татарстан": (55.796, 49.108),
            "Республика Башкортостан": (54.735, 55.958)
        }

        min_distance = float('inf')
        closest_region = None

        for region, center in region_centers.items():
            distance = ((lat - center[0]) ** 2 + (lon - center[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_region = region

        return closest_region if min_distance < 5 else None

    def set_click_handler(self, callback):
        try:
            if hasattr(self.map_widget, 'add_left_click_map_command'):
                self.map_widget.add_left_click_map_command(callback)
            else:
                self.map_widget.add_right_click_menu_command("Выбрать регион", callback, pass_coords=True)
        except Exception as e:
            print(f"Ошибка установки обработчика кликов: {e}")
            self.map_widget.canvas.bind("<Button-1>", lambda event: callback((event.x, event.y)))

    def center_on_russia(self):
        self.map_widget.set_position(65, 90)
        self.map_widget.set_zoom(3)


class PredictionDialog(tk.Toplevel):
    def __init__(self, parent, region_name, available_crops):
        super().__init__(parent)
        self.parent = parent
        self.region_name = region_name
        self.available_crops = available_crops
        self.result = None
        self.title(f"Прогноз урожайности - {region_name}")
        self.geometry("500x600")
        self.configure(bg=Config.COLORS['primary'])
        self.transient(parent)
        self.grab_set()
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text=f"Параметры прогноза\n{self.region_name}", font=("Arial", 14, "bold"),
                                justify=tk.CENTER)
        title_label.pack(pady=10)

        ttk.Label(main_frame, text="Сельскохозяйственная культура:", font=("Arial", 10, "bold")).pack(anchor=tk.W,
                                                                                                      pady=5)
        self.crop_var = tk.StringVar(value="Пшеница")
        crop_combo = ttk.Combobox(main_frame, textvariable=self.crop_var, values=list(self.available_crops.keys()),
                                  state="readonly", font=("Arial", 10))
        crop_combo.pack(fill=tk.X, pady=5)

        ttk.Label(main_frame, text="Период прогнозирования:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.period_var = tk.StringVar(value="1 год")
        period_combo = ttk.Combobox(main_frame, textvariable=self.period_var,
                                    values=list(Config.FORECAST_PERIODS.keys()), state="readonly", font=("Arial", 10))
        period_combo.pack(fill=tk.X, pady=5)

        ttk.Label(main_frame, text="Модель прогнозирования:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.model_var = tk.StringVar(value="Ensemble")
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var, values=list(Config.MODEL_TYPES.keys()),
                                   state="readonly", font=("Arial", 10))
        model_combo.pack(fill=tk.X, pady=5)

        params_frame = ttk.LabelFrame(main_frame, text="Дополнительные параметры")
        params_frame.pack(fill=tk.X, pady=10)

        self.weather_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Учитывать погодные данные", variable=self.weather_var).pack(anchor=tk.W,
                                                                                                        pady=2)

        self.economic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Учитывать экономические показатели", variable=self.economic_var).pack(
            anchor=tk.W, pady=2)

        self.historical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Учитывать исторические данные", variable=self.historical_var).pack(
            anchor=tk.W, pady=2)

        advanced_frame = ttk.LabelFrame(main_frame, text="Расширенные настройки")
        advanced_frame.pack(fill=tk.X, pady=10)

        ttk.Label(advanced_frame, text="Исторический период (лет):").pack(anchor=tk.W, pady=2)
        self.history_var = tk.IntVar(value=5)
        history_spin = ttk.Spinbox(advanced_frame, from_=3, to=10, textvariable=self.history_var, width=10)
        history_spin.pack(anchor=tk.W, pady=2)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Отмена", command=self.cancel).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сделать прогноз", command=self.confirm).pack(side=tk.LEFT, padx=5)

    def cancel(self):
        self.result = None
        self.destroy()

    def confirm(self):
        self.result = {
            'crop': self.crop_var.get(),
            'period': self.period_var.get(),
            'model': self.model_var.get(),
            'weather': self.weather_var.get(),
            'economic': self.economic_var.get(),
            'historical': self.historical_var.get(),
            'history_years': self.history_var.get()
        }
        self.destroy()


class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, region_name, prediction_data, historical_data, feature_importance):
        super().__init__(parent)
        self.parent = parent
        self.region_name = region_name
        self.prediction_data = prediction_data
        self.historical_data = historical_data
        self.feature_importance = feature_importance
        self.title(f"Результаты прогноза - {region_name}")
        self.geometry("1200x800")
        self.configure(bg=Config.COLORS['primary'])
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_summary_tab(notebook)
        self.create_charts_tab(notebook)
        self.create_analysis_tab(notebook)
        self.create_export_tab(notebook)

    def create_summary_tab(self, notebook):
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Сводка")

        main_info = ttk.LabelFrame(summary_frame, text="Основные результаты")
        main_info.pack(fill=tk.X, padx=10, pady=5)
        info_text = (f"Регион: {self.region_name}\nКультура: {self.prediction_data['crop']}\n"
                     f"Период прогноза: {self.prediction_data['period']}\nМодель: {self.prediction_data['model']}\n"
                     f"Дата расчета: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        ttk.Label(main_info, text=info_text, justify=tk.LEFT, font=("Arial", 10)).pack(padx=10, pady=10)

        metrics_frame = ttk.LabelFrame(summary_frame, text="Ключевые метрики")
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        metrics_text = (f"Прогнозируемая урожайность: {self.prediction_data['predicted_yield']:.1f} ц/га\n"
                        f"Изменение к прошлому году: {self.prediction_data['change']:+.1f}%\n"
                        f"Вероятность правильности: {self.prediction_data['confidence']:.1%}\n"
                        f"Доверительный интервал: ±{self.prediction_data['deviation']:.1f} ц/га")
        if 'model_quality' in self.prediction_data:
            metrics_text += f"\nКачество модели: {self.prediction_data['model_quality']:.1%}"
        ttk.Label(metrics_frame, text=metrics_text, justify=tk.LEFT, font=("Arial", 10)).pack(padx=10, pady=10)

        recommendations_frame = ttk.LabelFrame(summary_frame, text="Рекомендации")
        recommendations_frame.pack(fill=tk.X, padx=10, pady=5)
        recommendations = self.generate_recommendations()
        for rec in recommendations:
            ttk.Label(recommendations_frame, text=rec, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=2)

    def create_charts_tab(self, notebook):
        charts_frame = ttk.Frame(notebook)
        notebook.add(charts_frame, text="Графики")
        fig = Figure(figsize=(12, 8), dpi=100)
        ax1 = fig.add_subplot(221)
        self.plot_yield_trend(ax1)
        ax2 = fig.add_subplot(222)
        self.plot_year_comparison(ax2)
        ax3 = fig.add_subplot(223)
        self.plot_factors(ax3)
        ax4 = fig.add_subplot(224)
        self.plot_probability_distribution(ax4)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_analysis_tab(self, notebook):
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Анализ")
        factors_frame = ttk.LabelFrame(analysis_frame, text="Анализ факторов влияния")
        factors_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget = tk.Text(factors_frame, wrap=tk.WORD, width=80, height=20)
        scrollbar = ttk.Scrollbar(factors_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        analysis_report = self.generate_analysis_report()
        text_widget.insert(tk.END, analysis_report)
        text_widget.config(state=tk.DISABLED)

    def create_export_tab(self, notebook):
        export_frame = ttk.Frame(notebook)
        notebook.add(export_frame, text="Экспорт")
        ttk.Label(export_frame, text="Экспорт результатов прогноза", font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Button(export_frame, text="Excel файл с полным отчетом", command=self.export_to_excel).pack(pady=5)
        ttk.Button(export_frame, text="Сохранить графики", command=self.export_charts).pack(pady=5)

    def plot_yield_trend(self, ax):
        if self.historical_data and 'years' in self.historical_data and 'yields' in self.historical_data:
            years = self.historical_data['years']
            yields = self.historical_data['yields']
            if len(years) > 0 and len(yields) > 0:
                ax.plot(years, yields, 'bo-', label='Исторические данные', linewidth=2)
                if len(years) > 0:
                    last_year = years[-1]
                    ax.plot([last_year, last_year + 1], [yields[-1], self.prediction_data['predicted_yield']], 'ro--',
                            label='Прогноз', linewidth=2)
                    ax.fill_between([last_year + 1],
                                    self.prediction_data['predicted_yield'] - self.prediction_data['deviation'],
                                    self.prediction_data['predicted_yield'] + self.prediction_data['deviation'],
                                    alpha=0.3, color='red', label='Доверительный интервал')
        ax.set_title('Динамика урожайности')
        ax.set_xlabel('Год')
        ax.set_ylabel('Урожайность (ц/га)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_year_comparison(self, ax):
        if self.historical_data and 'yields' in self.historical_data and len(self.historical_data['yields']) > 0:
            categories = ['Прошлый год', 'Прогноз']
            values = [self.historical_data['yields'][-1], self.prediction_data['predicted_yield']]
            colors = [Config.COLORS['accent'], Config.COLORS['success']]
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_title('Сравнение с предыдущим годом')
            ax.set_ylabel('Урожайность (ц/га)')
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{value:.1f}', ha='center',
                        va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Недостаточно данных\nдля сравнения', ha='center', va='center', transform=ax.transAxes)

    def plot_factors(self, ax):
        if self.feature_importance:
            factors = list(self.feature_importance.keys())[:8]
            importance = list(self.feature_importance.values())[:8]
            y_pos = np.arange(len(factors))
            ax.barh(y_pos, importance, alpha=0.7, color=Config.COLORS['warning'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(factors)
            ax.set_title('Важность факторов влияния')
            ax.set_xlabel('Важность')
        else:
            ax.text(0.5, 0.5, 'Данные о важности факторов\nнедоступны', ha='center', va='center',
                    transform=ax.transAxes)

    def plot_probability_distribution(self, ax):
        mean = self.prediction_data['predicted_yield']
        std = self.prediction_data['deviation'] / 2
        x = np.linspace(max(0, mean - 3 * std), mean + 3 * std, 100)
        y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, y, 'g-', linewidth=2, label='Распределение вероятностей')
        ax.fill_between(x, y, alpha=0.3, color='green')
        ax.axvline(mean, color='red', linestyle='--', label=f'Среднее: {mean:.1f}')
        ax.set_title('Вероятностное распределение прогноза')
        ax.set_xlabel('Урожайность (ц/га)')
        ax.set_ylabel('Плотность вероятности')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def generate_recommendations(self):
        recommendations = []
        change = self.prediction_data['change']
        confidence = self.prediction_data['confidence']
        if change > 5:
            recommendations.append("• Благоприятный прогноз")
            recommendations.append("• Увеличить инвестиции в качественные семена")
        elif change < -5:
            recommendations.append("• Снижение урожайности - подготовить резервные планы")
            recommendations.append("• Увеличить мониторинг состояния посевов")
        else:
            recommendations.append("• Стабильная ситуация - поддерживать текущую стратегию")
        if confidence < 0.7:
            recommendations.append("• Низкая уверенность прогноза - усилить мониторинг")
        recommendations.append("• Регулярно обновлять данные для улучшения точности прогнозов")
        recommendations.append("• Сравнивать прогнозы с фактическими результатами для калибровки моделей")
        return recommendations

    def generate_analysis_report(self):
        report = "ДЕТАЛЬНЫЙ АНАЛИТИЧЕСКИЙ ОТЧЕТ\n" + "=" * 50 + "\n\n"
        report += f"РЕГИОН: {self.region_name}\nКУЛЬТУРА: {self.prediction_data['crop']}\n"
        report += f"ПЕРИОД ПРОГНОЗА: {self.prediction_data['period']}\nДАТА АНАЛИЗА: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
        report += "КЛЮЧЕВЫЕ МЕТРИКИ:\n"
        report += f"- Прогнозируемая урожайность: {self.prediction_data['predicted_yield']:.1f} ц/га\n"
        report += f"- Изменение к прошлому году: {self.prediction_data['change']:+.1f}%\n"
        report += f"- Вероятность правильности: {self.prediction_data['confidence']:.1%}\n"
        report += f"- Доверительный интервал: ±{self.prediction_data['deviation']:.1f} ц/га\n\n"
        report += "АНАЛИЗ ФАКТОРОВ ВЛИЯНИЯ:\n"
        if self.feature_importance:
            sorted_factors = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for factor, importance in sorted_factors[:10]:
                report += f"- {factor}: {importance:.3f}\n"
        else:
            report += "Данные о важности факторов недоступны\n"
        report += "\nРЕКОМЕНДАЦИИ:\n"
        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        return report

    def export_to_excel(self):
        filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                                                title="Сохранить отчет как...")
        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    summary_data = {
                        'Параметр': ['Регион', 'Культура', 'Период прогноза', 'Прогнозируемая урожайность (ц/га)',
                                     'Изменение (%)', 'Вероятность (%)', 'Отклонение (ц/га)', 'Модель прогнозирования',
                                     'Дата расчета'],
                        'Значение': [self.region_name, self.prediction_data['crop'], self.prediction_data['period'],
                                     f"{self.prediction_data['predicted_yield']:.2f}",
                                     f"{self.prediction_data['change']:+.2f}",
                                     f"{self.prediction_data['confidence'] * 100:.1f}",
                                     f"{self.prediction_data['deviation']:.2f}", self.prediction_data['model'],
                                     datetime.now().strftime('%d.%m.%Y %H:%M')]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Основные результаты', index=False)
                    if self.feature_importance:
                        factors_df = pd.DataFrame({'Фактор': list(self.feature_importance.keys()),
                                                   'Важность': list(self.feature_importance.values())}).sort_values(
                            'Важность', ascending=False)
                        factors_df.to_excel(writer, sheet_name='Факторы влияния', index=False)
                    if self.historical_data and 'years' in self.historical_data and 'yields' in self.historical_data:
                        history_df = pd.DataFrame(
                            {'Год': self.historical_data['years'], 'Урожайность': self.historical_data['yields']})
                        history_df.to_excel(writer, sheet_name='Исторические данные', index=False)
                messagebox.showinfo("Успех", f"Отчет успешно экспортирован в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось экспортировать файл:\n{str(e)}")

    def export_charts(self):
        try:
            fig = Figure(figsize=(12, 8), dpi=100)
            ax1 = fig.add_subplot(221)
            self.plot_yield_trend(ax1)
            ax2 = fig.add_subplot(222)
            self.plot_year_comparison(ax2)
            ax3 = fig.add_subplot(223)
            self.plot_factors(ax3)
            ax4 = fig.add_subplot(224)
            self.plot_probability_distribution(ax4)
            fig.tight_layout()
            filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                                                    title="Сохранить графики как...")
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"Графики сохранены в:\n{filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить графики:\n{str(e)}")


class AgriculturalPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌾 Продвинутая система прогнозирования урожайности")
        self.root.geometry("1400x900")
        self.root.configure(bg=Config.COLORS['primary'])

        self.status_var = tk.StringVar(value="Система инициализируется...")

        self.data_loader = AgriculturalDataLoader()
        self.predictor = AdvancedYieldPredictor()
        self.map_handler = None
        self.data_loaded = False
        self.model_trained = False
        self.current_region = None
        self.borders_loaded = False

        self.setup_styles()
        self.create_widgets()
        self.load_data_async()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=Config.COLORS['primary'])
        style.configure('TLabel', background=Config.COLORS['primary'], foreground='white')
        style.configure('TButton', background=Config.COLORS['accent'], foreground='white')
        style.configure('TLabelframe', background=Config.COLORS['secondary'], foreground='white')
        style.configure('TLabelframe.Label', background=Config.COLORS['secondary'], foreground='white')
        style.configure('TNotebook', background=Config.COLORS['primary'])
        style.configure('TNotebook.Tab', background=Config.COLORS['secondary'], foreground='white')

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_header(main_frame)
        self.create_control_panel(main_frame)
        self.create_main_content(main_frame)
        self.create_status_bar(main_frame)

    def create_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = tk.Label(header_frame, text="🌾 ПРОДВИНУТАЯ СИСТЕМА ПРОГНОЗИРОВАНИЯ УРОЖАЙНОСТИ",
                               font=("Arial", 18, "bold"), fg="white", bg=Config.COLORS['primary'])
        title_label.pack()
        subtitle_label = tk.Label(header_frame,
                                  text="Нейросетевое прогнозирование с учетом погодных, экономических и исторических факторов",
                                  font=("Arial", 10), fg="#bdc3c7", bg=Config.COLORS['primary'])
        subtitle_label.pack()

    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Управление системой")
        control_frame.pack(fill=tk.X, pady=10)

        top_row = ttk.Frame(control_frame)
        top_row.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(top_row, text="📁 Загрузить данные", command=self.load_data_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_row, text="🗺️ Загрузить границы", command=self.load_borders_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_row, text="🌾 Обучить модели", command=self.train_models_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_row, text="🔄 Обновить данные", command=self.refresh_data).pack(side=tk.LEFT, padx=5)

        bottom_row = ttk.Frame(control_frame)
        bottom_row.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(bottom_row, text="🗺️ Показать все границы", command=self.show_all_borders).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_row, text="📊 Статистика", command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_row, text="📍 Центрировать карту", command=self.center_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_row, text="🗑️ Очистить карту", command=self.clear_map).pack(side=tk.LEFT, padx=5)

    def create_main_content(self, parent):
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        map_frame = ttk.LabelFrame(content_frame, text="Интерактивная карта России")
        map_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))

        self.map_handler = MapHandler(map_frame)
        self.map_handler.set_click_handler(self.on_map_click)

        self.load_borders_on_startup()

        info_frame = ttk.LabelFrame(content_frame, text="Информация и управление")
        info_frame.pack(fill=tk.BOTH, expand=False, side=tk.RIGHT, padx=(5, 0), ipadx=10)

        self.create_info_panel(info_frame)

    def create_info_panel(self, parent):
        status_frame = ttk.LabelFrame(parent, text="Статус системы")
        status_frame.pack(fill=tk.X, pady=5)

        self.status_text = tk.Text(status_frame, height=8, width=40, bg=Config.COLORS['light'], wrap=tk.WORD,
                                   font=("Arial", 9))
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)

        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        quick_actions = ttk.LabelFrame(parent, text="Быстрое управление")
        quick_actions.pack(fill=tk.X, pady=5)

        ttk.Button(quick_actions, text="🚀 Быстрый прогноз", command=self.quick_prediction).pack(fill=tk.X, padx=5,
                                                                                                pady=2)
        ttk.Button(quick_actions, text="📋 История прогнозов", command=self.show_prediction_history).pack(fill=tk.X,
                                                                                                         padx=5, pady=2)

        self.region_info = ttk.LabelFrame(parent, text="Информация о регионе")
        self.region_info.pack(fill=tk.X, pady=5)

        self.region_label = ttk.Label(self.region_info, text="Регион не выбран", font=("Arial", 10, "bold"))
        self.region_label.pack(padx=10, pady=10)

    def create_status_bar(self, parent):
        status_bar = ttk.Frame(parent)
        status_bar.pack(fill=tk.X, pady=5)

        status_label = ttk.Label(status_bar, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=2)

    def load_borders_on_startup(self):
        def load_task():
            self.update_status("🗺️ Загрузка границ регионов...")
            try:
                count = self.map_handler.load_regions_data()
                if count > 0:
                    self.borders_loaded = True
                    self.update_status(f"✅ Границы загружены: {count} регионов")
                else:
                    self.update_status("⚠️ Границы не загружены, используйте загрузку вручную")
            except Exception as e:
                self.update_status(f"❌ Ошибка загрузки границ: {e}")

        thread = threading.Thread(target=load_task)
        thread.daemon = True
        thread.start()

    def load_borders_dialog(self):
        filetypes = [
            ("JSON files", "*.json"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Выберите файл с границами регионов",
            filetypes=filetypes
        )

        if filename:
            self.update_status(f"🗺️ Загрузка границ из {os.path.basename(filename)}...")
            try:
                count = self.map_handler.load_regions_data(filename)
                if count > 0:
                    self.borders_loaded = True
                    self.update_status(f"✅ Границы загружены: {count} регионов")
                    self.update_info(f"Загружены границы {count} регионов\n\n"
                                     "Границы можно отобразить через:\n"
                                     "'Показать все границы' или\n"
                                     "выбрав регион на карте")
                else:
                    self.update_status("❌ Не удалось загрузить границы")
            except Exception as e:
                self.update_status(f"❌ Ошибка загрузки границ: {e}")

    def show_all_borders(self):
        if not self.borders_loaded:
            messagebox.showwarning("Внимание", "Сначала загрузите границы регионов!")
            return

        self.map_handler.show_all_regions_borders()
        self.update_status("🗺️ Отображены границы всех регионов")

    def load_data_async(self):
        def load_task():
            self.update_status("🔄 Загрузка данных...")
            try:
                success = self.data_loader.load_all_data()
                self.data_loaded = success
                if success:
                    self.update_status("✅ Данные успешно загружены")
                    available_regions = self.data_loader.get_available_regions()
                    regions_info = "\n".join(available_regions[:10])
                    self.update_info(
                        f"Система готова к работе!\n\nДоступные регионы ({len(available_regions)}):\n{regions_info}" +
                        ("\n..." if len(available_regions) > 10 else "") +
                        "\n\nДля начала работы:\n1. Выберите регион на карте\n2. Настройте параметры прогноза\n3. Запустите расчет")
                else:
                    self.update_status("❌ Ошибка загрузки данных")
                    self.update_info(
                        "Не удалось загрузить данные.\nПроверьте наличие файлов в папке agricultural_data/")
            except Exception as e:
                self.update_status("❌ Ошибка загрузки данных")
                self.update_info(f"Ошибка загрузки: {str(e)}\nПроверьте структуру данных и файлы.")

        thread = threading.Thread(target=load_task)
        thread.daemon = True
        thread.start()

    def load_data_dialog(self):
        filetypes = [("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Выберите файл с данными", filetypes=filetypes)
        if filename:
            self.update_status(f"🔄 Загрузка {os.path.basename(filename)}...")
            self.update_status("✅ Файл загружен")

    def train_models_dialog(self):
        if not self.data_loaded:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        self.update_status("🧠 Обучение моделей...")

        all_X = None
        all_y = None
        regions = self.data_loader.get_available_regions()
        trained_regions = 0

        selected_regions = []

        weather_regions_count = 0
        for region in regions:
            region_data = self.data_loader.get_region_data(region)
            if region_data and region_data.get('weather') is not None:
                selected_regions.append(region)
                weather_regions_count += 1
                if weather_regions_count >= 3:
                    break

        for region in regions:
            if region not in selected_regions and len(selected_regions) < 5:
                region_data = self.data_loader.get_region_data(region)
                if region_data and region_data['yield'] is not None and len(region_data['yield']) > 3:
                    selected_regions.append(region)

        print(f"🎯 Регионы для обучения: {selected_regions}")

        for region in selected_regions:
            try:
                region_data = self.data_loader.get_region_data(region)
                if region_data and region_data['yield'] is not None and len(region_data['yield']) > 3:
                    X, y = self.predictor.prepare_features(region_data)

                    if X is not None and not X.empty:
                        if all_X is None:
                            all_X = X
                            all_y = y
                        else:
                            all_X = pd.concat([all_X, X], ignore_index=True)
                            all_y = pd.concat([all_y, y], ignore_index=True)
                        trained_regions += 1

                        weather_status = "с погодными данными" if region_data.get(
                            'weather') is not None else "без погодных данных"
                        print(f"✅ Добавлены данные региона: {region} ({weather_status})")

                        if region_data.get('weather') is not None:
                            print(
                                f"   📊 Доступные погодные признаки: {[col for col in X.columns if any(weather_term in col for weather_term in ['temp', 'pressure', 'humidity'])]}")

                        if region_data.get('mosbir_index') is not None and not region_data['mosbir_index'].empty:
                            print(
                                f"   📈 Доступные признаки индекса МосБиржи: {[col for col in X.columns if 'mosbir' in col]}")
            except Exception as e:
                print(f"Ошибка обработки региона {region}: {e}")

        if all_X is not None and not all_X.empty:
            print(f"📊 Всего признаков в обучающей выборке: {len(all_X.columns)}")
            print(f"📋 Используемые признаки: {all_X.columns.tolist()}")

            weather_features = [col for col in all_X.columns if any(
                weather_term in col for weather_term in ['temp', 'pressure', 'humidity', 'spring', 'summer'])]
            print(f"🌤️  Погодные признаки в данных: {weather_features}")

            mosbir_features = [col for col in all_X.columns if 'mosbir' in col]
            print(f"📈 Признаки индекса МосБиржи в данных: {mosbir_features}")

            success = self.predictor.train_models(all_X, all_y)
            if success:
                self.model_trained = True
                self.update_status("✅ Модели успешно обучены")
                perf = self.predictor.get_model_performance()
                perf_text = "\n".join([f"{k}: R²={v['r2']:.3f}, MAE={v['mae']:.3f}" for k, v in perf.items()])

                feature_importance = self.predictor.calculate_feature_importance(all_X, all_X.columns)
                if feature_importance:
                    top_features = "\n".join(
                        [f"- {feat}: {imp:.3f}" for feat, imp in list(feature_importance.items())[:5]])
                    importance_info = f"\n\nТоп-5 важных признаков:\n{top_features}"
                else:
                    importance_info = "\n\nНе удалось рассчитать важность признаков"

                self.update_info(
                    f"Модели обучены на {trained_regions} регионах\n\nПроизводительность моделей:\n{perf_text}{importance_info}")
            else:
                self.update_status("❌ Ошибка обучения моделей")
                self.update_info("Не удалось обучить модели. Проверьте качество данных.")
        else:
            self.update_status("❌ Недостаточно данных для обучения")
            self.update_info("Недостаточно данных для обучения моделей.")

    def on_map_click(self, coords):
        lat, lon = coords

        marker = self.map_handler.add_marker(lat, lon, "Выбранная точка")
        if marker is None:
            print("❌ Не удалось добавить маркер на карту")
            return

        region_name = self.map_handler.find_region_by_coords(lat, lon)
        self.current_region = region_name

        if region_name:
            normalized_region = str(region_name).strip()
            self.region_label.config(text=normalized_region)

            if self.borders_loaded:
                self.map_handler.highlight_region(normalized_region)

            available_regions = self.data_loader.get_available_regions()

            region_found = False
            for available_region in available_regions:
                if (normalized_region.lower() in available_region.lower() or
                        available_region.lower() in normalized_region.lower()):
                    region_found = True
                    self.current_region = available_region
                    break

            if region_found:
                region_data = self.data_loader.get_region_data(self.current_region)
                weather_info = "Доступны ✓" if region_data and region_data.get('weather') is not None else "Нет данных"
                yield_info = "Доступны ✓" if region_data and not region_data['yield'].empty else "Нет данных"
                mosbir_info = "Доступны ✓" if region_data and region_data.get('mosbir_index') is not None and not \
                    region_data['mosbir_index'].empty else "Нет данных"

                self.update_info(f"📍 Выбран регион: {self.current_region}\n\n"
                                 f"Координаты: {lat:.4f}, {lon:.4f}\n"
                                 f"Границы: {'Загружены ✓' if self.borders_loaded else 'Не загружены'}\n"
                                 f"Данные урожайности: {yield_info}\n"
                                 f"Погодные данные: {weather_info}\n"
                                 f"Индекс МосБиржи: {mosbir_info}\n"
                                 f"Готов к прогнозированию!")
            else:
                self.update_info(f"📍 Выбран регион: {normalized_region}\n\n"
                                 f"Координаты: {lat:.4f}, {lon:.4f}\n"
                                 f"Границы: {'Загружены ✓' if self.borders_loaded else 'Не загружены'}\n"
                                 f"❌ Данные урожайности недоступны\n"
                                 f"Выберите другой регион")
        else:
            self.region_label.config(text="Регион не определен")
            self.update_info("Не удалось определить регион.\n"
                             "Попробуйте выбрать точку ближе\nк центру региона или загрузите границы.")

    def quick_prediction(self):
        if not self.current_region:
            messagebox.showwarning("Внимание", "Сначала выберите регион на карте!")
            return
        if not self.model_trained:
            messagebox.showwarning("Внимание", "Сначала обучите модели!")
            return

        region_data = self.data_loader.get_region_data(self.current_region)
        if region_data is None or region_data['yield'] is None or region_data['yield'].empty:
            messagebox.showerror("Ошибка", f"Нет данных урожайности для региона {self.current_region}")
            return

        dialog = PredictionDialog(self.root, self.current_region, Config.CROPS)
        self.root.wait_window(dialog)
        if dialog.result:
            self.run_prediction(dialog.result)

    def run_prediction(self, parameters):
        self.update_status("🎯 Расчет прогноза...")
        try:
            region_data = self.data_loader.get_region_data(self.current_region)
            if region_data is None or region_data['yield'] is None:
                messagebox.showerror("Ошибка", f"Нет данных для региона {self.current_region}")
                return

            X, y = self.predictor.prepare_features(region_data, lookback_period=parameters['history_years'])
            if X is None or X.empty:
                messagebox.showerror("Ошибка", "Недостаточно данных для прогноза")
                return

            last_X = X.iloc[[-1]]
            prediction, confidence, deviation = self.predictor.predict(last_X, parameters['model'])
            if prediction is None:
                messagebox.showerror("Ошибка", "Не удалось рассчитать прогноз")
                return

            last_yield = region_data['yield']['yield'].iloc[-1] if len(region_data['yield']) > 0 else 0
            change = ((prediction - last_yield) / last_yield * 100) if last_yield > 0 else 0
            prediction_data = {
                'predicted_yield': prediction, 'confidence': confidence, 'deviation': deviation,
                'change': change, 'crop': parameters['crop'], 'period': parameters['period'],
                'model': parameters['model']
            }

            historical_data = {
                'years': region_data['yield']['year'].tolist(),
                'yields': region_data['yield']['yield'].tolist()
            }

            feature_importance = self.predictor.calculate_feature_importance(X, self.predictor.feature_names)
            ResultsWindow(self.root, self.current_region, prediction_data, historical_data, feature_importance)
            self.update_status("✅ Прогноз готов!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при расчете прогноза: {str(e)}")
            self.update_status("❌ Ошибка прогнозирования")

    def update_status(self, message):
        def safe_update():
            self.status_var.set(message)
            self.root.update_idletasks()

        self.root.after(0, safe_update)

    def update_info(self, message):
        def safe_update():
            self.status_text.config(state=tk.NORMAL)
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, message)
            self.status_text.config(state=tk.DISABLED)

        self.root.after(0, safe_update)

    def refresh_data(self):
        self.load_data_async()

    def show_statistics(self):
        if self.data_loaded:
            stats = f"Статистика системы:\n\nРегионов: {len(self.data_loader.yield_data['region'].unique())}\n"
            stats += f"Городов с погодой: {len(self.data_loader.weather_data)}\n"
            stats += f"Моделей обучено: {len(self.predictor.models) if self.model_trained else 0}\n"
            stats += f"Границы регионов: {'Загружены' if self.borders_loaded else 'Не загружены'}\n"

            if self.data_loader.mosbir_index_data is not None and not self.data_loader.mosbir_index_data.empty:
                stats += f"Записей индекса МосБиржи: {len(self.data_loader.mosbir_index_data)}\n"
                min_date = self.data_loader.mosbir_index_data['date'].min()
                max_date = self.data_loader.mosbir_index_data['date'].max()
                stats += f"Диапазон дат индекса: {min_date.strftime('%d.%m.%Y')} - {max_date.strftime('%d.%m.%Y')}\n"

            if self.borders_loaded:
                stats += f"Регионов с границами: {len(self.map_handler.loaded_regions)}\n"

            if self.predictor.training_history:
                last_train = self.predictor.training_history[-1]
                stats += f"\nПоследнее обучение:\n"
                for model, score in last_train['performance'].items():
                    stats += f"- {model}: R²={score['r2']:.3f}, MAE={score['mae']:.3f}\n"

            messagebox.showinfo("Статистика", stats)
        else:
            messagebox.showwarning("Внимание", "Данные не загружены!")

    def center_map(self):
        if self.map_handler:
            self.map_handler.center_on_russia()

    def clear_map(self):
        if self.map_handler:
            self.map_handler.clear_map()
            self.current_region = None
            self.region_label.config(text="Регион не выбран")
            self.update_status("🗑️ Карта очищена")
            self.update_info("Карта очищена.\nВыберите новый регион на карте.")

    def show_prediction_history(self):
        messagebox.showinfo("История", "История прогнозов в разработке")


def main():
    try:
        import tkintermapview
        import matplotlib.pyplot as plt
        import pandas as pd
        import sklearn
        import xgboost as xgb
        import lightgbm as lgb

        root = tk.Tk()
        app = AgriculturalPredictorApp(root)
        root.mainloop()

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        messagebox.showerror("Ошибка", f"Не удалось запустить приложение: {e}")


if __name__ == "__main__":
    main()