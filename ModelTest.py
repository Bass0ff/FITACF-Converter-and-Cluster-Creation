import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
from glob import glob

# ===================================================================
# ЖЁСТКО ЗАДАННЫЙ СПИСОК ПРИЗНАКОВ
# ===================================================================
FEATURE_COLS = ['vel', 'width', 'height_out', 'hop_out', 'kvert_cos', 'kb_cos', 
                'kvert_cos_half', 'kvert_cos_q1', 'kvert_cos_q3']

# ===================================================================
# ПАРСЕР .fit ФАЙЛОВ
# ===================================================================
def parse_fit_file(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 24:
                continue
            try:
                row = {
                    'year': int(parts[0]), 'month': int(parts[1]), 'day': int(parts[2]),
                    'time_hours': float(parts[3]), 'range_km': float(parts[4]),
                    'channel': int(parts[5]), 'tfreq': int(parts[6]), 'beam': int(parts[7]),
                    'power': float(parts[8]), 'vel': float(parts[9]), 'width': float(parts[10]),
                    'elv_normal': float(parts[11]), 'height_simple': float(parts[12]),
                    'height_out': float(parts[13]), 'hop_out': int(parts[14]),
                    'kvert_cos': float(parts[15]), 'kb_cos': float(parts[16]),
                    'kvert_cos_half': float(parts[17]), 'kvert_cos_q1': float(parts[18]),
                    'kvert_cos_q3': float(parts[19]), 'gflg': int(parts[20]),
                    'elv_low': int(parts[21]), 'mplgs': int(parts[22]), 'mppul': int(parts[23]),
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)

# ===================================================================
# ЗАГРУЗКА МОДЕЛИ
# ===================================================================
def load_model(model_path="xgboost_model_for_inference.json"):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

# ===================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ===================================================================
def classify_file(input_path, output_path, model_path="xgboost_model_for_inference.json"):
    model = load_model(model_path)

    # Определяем, что дали: папку, .fit или .csv
    if os.path.isdir(input_path):
        fit_files = sorted(glob(os.path.join(input_path, "*.fit")))
        if not fit_files:
            print(f"ОШИБКА: В папке {input_path} не найдено .fit файлов")
            return
        print(f"Найдено {len(fit_files)} .fit файлов")
        dfs = []
        for f in fit_files:
            print(f"  Парсинг: {os.path.basename(f)}")
            dfs.append(parse_fit_file(f))
        df = pd.concat(dfs, ignore_index=True)
        print(f"Всего распаршено {len(df)} записей")
    elif input_path.endswith('.fitacf'):
        print(f"Парсинг: {input_path}")
        df = parse_fit_file(input_path)
        print(f"Распаршено {len(df)} записей")
    elif input_path.endswith('.csv'):
        print(f"Читаю CSV: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Загружено {len(df)} записей")
    else:
        print(f"ОШИБКА: Неизвестный формат {input_path}")
        return

    # Проверяем признаки
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ОШИБКА: Отсутствуют колонки: {missing}")
        return

    # Предсказания
    X = df[FEATURE_COLS]
    df['prediction'] = model.predict(X)
    proba = model.predict_proba(X)
    df['prob_IS'] = proba[:, 0]
    df['prob_GS'] = proba[:, 1]

    n_is = (df['prediction'] == 0).sum()
    n_gs = (df['prediction'] == 1).sum()

    df.to_csv(output_path, index=False)
    print(f"\nГотово: {output_path}")
    print(f"IS: {n_is} ({n_is/len(df)*100:.1f}%)")
    print(f"GS: {n_gs} ({n_gs/len(df)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        classify_file(sys.argv[1], sys.argv[2])
    else:
        print("Использование: python classify_radar.py <входной_файл_или_папка> <выходной_файл.csv>")
        print("Примеры:")
        print("  python classify_radar.py data.fit output.csv")
        print("  python classify_radar.py ./папка_с_fit/ output.csv")