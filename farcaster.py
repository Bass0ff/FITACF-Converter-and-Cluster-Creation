import pandas as pd
import numpy as np
import xgboost as xgb

# Загружаем датасет
print("Загружаю датасет...")
df = pd.read_csv("./clustering_superdarn_fork/dataset_for_ml_labeled.csv")

feature_cols = ['vel', 'width', 'height_out', 'hop_out', 'kvert_cos', 'kb_cos', 
                'kvert_cos_half', 'kvert_cos_q1', 'kvert_cos_q3']

# Загружаем модель
print("Загружаю модель...")
model = xgb.XGBClassifier()
model.load_model("xgboost_model_for_inference.json")

# Предсказания
X = df[feature_cols]
df['predicted'] = model.predict(X)

# Фильтруем ближние дальности (до 500 км)
df_near = df[df['range_km'] <= 500]

# Считаем проценты
total_near = len(df_near)
is_near_ribiero = (df_near['label'] == 0).sum()
is_near_model = (df_near['predicted'] == 0).sum()

print(f"\n{'='*50}")
print(f"АНАЛИЗ БЛИЖНИХ ДАЛЬНОСТЕЙ (до 500 км)")
print(f"{'='*50}")
print(f"Всего точек на дальностях до 500 км: {total_near}")
print(f"\nРибейро:")
print(f"  IS: {is_near_ribiero} ({is_near_ribiero/total_near*100:.1f}%)")
print(f"  GS: {total_near - is_near_ribiero} ({(total_near - is_near_ribiero)/total_near*100:.1f}%)")
print(f"\nМодель XGBoost:")
print(f"  IS: {is_near_model} ({is_near_model/total_near*100:.1f}%)")
print(f"  GS: {total_near - is_near_model} ({(total_near - is_near_model)/total_near*100:.1f}%)")