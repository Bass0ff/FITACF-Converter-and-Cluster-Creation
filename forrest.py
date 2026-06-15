import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from matplotlib.dates import num2date

# ===================================================================
# ФУНКЦИЯ РИБЕЙРО
# ===================================================================
def ribeiro_gs_flg(vel, time):
    if len(time) == 0:
        return True
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > 15.0)
    low = np.sum(np.abs(vel) <= 15.0)
    R = 1.0 if low == 0 else high / low
    if L > 14.0: return True if R <= 0.15 else False
    elif L > 3.0: return True if R <= 0.2 else False
    elif L > 2.0: return True if R <= 0.33 else False
    elif L > 1.0: return True if R <= 0.475 else False
    else: return True if R <= 0.5 else False

# ===================================================================
# ШАГ 1: Готовим разметку Рибейро (как обычно)
# ===================================================================
print("Шаг 1: Готовлю разметку Рибейро...")
data_dir = "./clustering_superdarn_fork/data/"
all_labels = []

for filepath in glob(os.path.join(data_dir, "*_dbgmm.csv")):
    try:
        df_ribiero = pd.read_csv(filepath, index_col=0)
        for scan_idx, row in df_ribiero.iterrows():
            try:
                gates = np.fromstring(row['gate'].strip('[]'), sep=' ')
                beams = np.fromstring(row['beam'].strip('[]'), sep=' ')
                vels = np.fromstring(row['vel'].strip('[]'), sep=' ')
                times_datenum = np.fromstring(row['time'].strip('[]'), sep=' ')
                clust_flags = np.fromstring(row['clust_flg'].strip('[]'), sep=' ')
                
                first_date = num2date(times_datenum[0])
                year, month, day = first_date.year, first_date.month, first_date.day
                time_hours = np.array([t.hour + t.minute/60.0 + t.second/3600.0 
                                      for t in num2date(times_datenum)])
                
                ribiero_labels = np.full_like(clust_flags, -1, dtype=int)
                for cluster_id in np.unique(clust_flags):
                    if cluster_id == -1: continue
                    mask = clust_flags == cluster_id
                    is_gs = ribeiro_gs_flg(vels[mask], times_datenum[mask])
                    ribiero_labels[mask] = 1 if is_gs else 0
                
                for i in range(len(gates)):
                    if ribiero_labels[i] != -1:
                        all_labels.append({
                            'year': year, 'month': month, 'day': day,
                            'time_hours': round(time_hours[i], 6),
                            'beam': int(beams[i]),
                            'label': ribiero_labels[i]
                        })
            except:
                pass
    except:
        pass

df_labels = pd.DataFrame(all_labels)
print(f"Размечено точек: {len(df_labels)}")

# ===================================================================
# ШАГ 2: Готовим словарь с метками для быстрого поиска
# ===================================================================
print("\nШаг 2: Строю словарь с метками...")
# Создаём ключи вида "YYYY_MM_DD_time_beam" для мгновенного поиска
df_labels['key'] = (df_labels['year'].astype(str) + '_' + 
                    df_labels['month'].astype(str).str.zfill(2) + '_' + 
                    df_labels['day'].astype(str).str.zfill(2) + '_' + 
                    df_labels['time_hours'].round(4).astype(str) + '_' + 
                    df_labels['beam'].astype(str))

# Создаём словарь: ключ -> метка
label_dict = dict(zip(df_labels['key'], df_labels['label']))
# Освобождаем память
del df_labels
print(f"Словарь создан, {len(label_dict)} уникальных ключей")

# ===================================================================
# ШАГ 3: Обрабатываем features.csv чанками
# ===================================================================
print("\nШаг 3: Сливаю данные по чанкам...")
CHUNK_SIZE = 50000
merged_chunks = []
total_rows = 0

reader = pd.read_csv("features.csv", chunksize=CHUNK_SIZE)
for chunk_num, chunk in enumerate(reader):
    # Создаём ключи для текущего чанка
    chunk['key'] = (chunk['year'].astype(str) + '_' + 
                    chunk['month'].astype(str).str.zfill(2) + '_' + 
                    chunk['day'].astype(str).str.zfill(2) + '_' + 
                    chunk['time_hours'].round(4).astype(str) + '_' + 
                    chunk['beam'].astype(str))
    
    # Находим метки через словарь (мгновенно!)
    chunk['label'] = chunk['key'].map(label_dict)
    
    # Оставляем только те строки, для которых нашлась метка
    chunk = chunk.dropna(subset=['label'])
    chunk['label'] = chunk['label'].astype(int)
    
    if len(chunk) > 0:
        merged_chunks.append(chunk.drop(columns=['key']))
        total_rows += len(chunk)
    
    if (chunk_num + 1) % 20 == 0:
        print(f"  Обработано {(chunk_num + 1) * CHUNK_SIZE} строк, найдено {total_rows} совпадений...")

df_merged = pd.concat(merged_chunks, ignore_index=True)
print(f"Объединённый датасет: {len(df_merged)} записей")

# ===================================================================
# ШАГ 4: Обучение Random Forest
# ===================================================================
feature_cols = ['vel', 'width', 'height_out', 'hop_out', 'kvert_cos', 'kb_cos', 
                'kvert_cos_half', 'kvert_cos_q1', 'kvert_cos_q3']
X = df_merged[feature_cols]
y = df_merged['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nОбучаю Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\n" + "="*50)
print("РЕЗУЛЬТАТ RANDOM FOREST")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['IS (0)', 'GS (1)']))

df_merged.to_csv("dataset_for_ml_labeled.csv", index=False)
print("\nДатасет сохранён в dataset_for_ml_labeled.csv")