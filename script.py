import pandas as pd
import numpy as np
from glob import glob
import os
import re

def parse_fit_file(filepath):
    """
    Парсит один .fit файл (который на самом деле текстовый)
    и возвращает pandas DataFrame с правильными колонками.
    """
    rows = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Пропускаем пустые строки
            if not line.strip():
                continue
            
            # ============================================================
            # ШАГ 1: Разбиваем строку по табуляции и пробелам
            # ============================================================
            # В данных есть и табы, и пробелы, поэтому используем
            # универсальный split(), который разбивает по любым пробельным символам
            parts = line.split()
            
            # Проверяем, что строка не пустая и не короче ожидаемой
            if len(parts) < 24:
                continue  # пропускаем битые строки
            
            # ============================================================
            # ШАГ 2: Извлекаем нужные нам колонки по номерам
            # ============================================================
            # Нумерация с 0:
            # parts[0]  = year
            # parts[1]  = month
            # parts[2]  = day
            # parts[3]  = time_hours
            # parts[4]  = range_gate_km
            # parts[5]  = channel
            # parts[6]  = tfreq
            # parts[7]  = bmnum
            # parts[8]  = p_l (мощность)
            # parts[9]  = v (скорость) - input1
            # parts[10] = w_l (ширина) - input2
            # parts[11] = elv_normal
            # parts[12] = height_simple
            # parts[13] = height_out (input3) ??? - нужно проверить
            # parts[14] = hop_out (input4)      ??? - нужно проверить
            # parts[15] = kvert_cos (input5)    ??? - нужно проверить
            # parts[16] = kb_cos (input6)       ??? - нужно проверить
            # parts[17] = kvert_cos_halfpath (input7)
            # parts[18] = kvert_cos_quater1path (input8)
            # parts[19] = kvert_cos_quater3path (input9)
            # parts[20] = gflg (флаг GS/IS из FITACF)
            # parts[21] = elv_low
            # parts[22] = mplgs (константа 23)
            # parts[23] = mppul (константа 8)
            
            try:
                row = {
                    'year': int(parts[0]),
                    'month': int(parts[1]),
                    'day': int(parts[2]),
                    'time_hours': float(parts[3]),
                    'range_km': float(parts[4]),
                    'channel': int(parts[5]),
                    'tfreq': int(parts[6]),
                    'beam': int(parts[7]),
                    'power': float(parts[8]),
                    'vel': float(parts[9]),         # input1
                    'width': float(parts[10]),      # input2
                    'elv_normal': float(parts[11]),
                    'height_simple': float(parts[12]),
                    'height_out': float(parts[13]),  # input3
                    'hop_out': int(parts[14]),       # input4
                    'kvert_cos': float(parts[15]),   # input5
                    'kb_cos': float(parts[16]),      # input6
                    'kvert_cos_half': float(parts[17]), # input7
                    'kvert_cos_q1': float(parts[18]),   # input8
                    'kvert_cos_q3': float(parts[19]),   # input9
                    'gflg': int(parts[20]),
                    'elv_low': int(parts[21]),
                    'mplgs': int(parts[22]),
                    'mppul': int(parts[23]),
                }
                rows.append(row)
            except (ValueError, IndexError) as e:
                # Если какое-то поле не распарсилось, пропускаем строку
                print(f"Warning: couldn't parse line in {filepath}: {line[:100]}...")
                continue
    
    # Создаём DataFrame
    df = pd.DataFrame(rows)
    return df


def load_all_data(data_dir="data.out", radar="cve", year=2020):
    """
    Загружает все .fit файлы из указанной директории.
    
    Параметры:
    -----------
    data_dir : str
        Путь к папке с выходными данными (data.out)
    radar : str
        Код радара (например, 'cve')
    year : int
        Год, за который обрабатывались данные
    
    Возвращает:
    -----------
    pd.DataFrame с объединёнными данными за все дни
    """
    # Ищем все .fit файлы для данного радара
    pattern = os.path.join(data_dir, radar, str(year), "*.fitacf")
    files = glob(pattern)
    
    if not files:
        # Пробуем другой шаблон (иногда файлы лежат без подпапок)
        pattern = os.path.join(data_dir, "*.fitacf")
        files = glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"Не найдено .fitacf файлов по шаблону {pattern}")
    
    print(f"Найдено {len(files)} файлов")
    
    all_dfs = []
    for i, fpath in enumerate(files):
        print(f"Обрабатываю {i+1}/{len(files)}: {os.path.basename(fpath)}", end='\r')
        df = parse_fit_file(fpath)
        if len(df) > 0:
            all_dfs.append(df)
    
    print(f"\nУспешно загружено {len(all_dfs)} файлов")
    
    # Объединяем все DataFrame в один
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Всего записей: {len(full_df)}")
        return full_df
    else:
        return pd.DataFrame()


# ===================================================================
# ЗАПУСК ПАРСИНГА
# ===================================================================
if __name__ == "__main__":
    # Загружаем данные
    df = load_all_data(
        data_dir="cve",  # папка, куда программа выгрузила обработанные файлы
        radar="cve",
        year=2021
    )
    
    # Сохраняем результат в CSV
    df.to_csv("features_2020.csv", index=False)
    print("Сохранено в features_2020.csv")
    
    # Выводим статистику
    print("\nПервые 5 строк:")
    print(df.head())
    print(f"\nРазмер датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")