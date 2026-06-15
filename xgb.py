import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils import resample

pd.options.mode.chained_assignment = None

print("Загружаю датасет...")
df = pd.read_csv("dataset_for_ml_labeled.csv")
print(f"Загружено {len(df)} записей")

feature_cols = ['vel', 'width', 'height_out', 'hop_out', 'kvert_cos', 'kb_cos', 
                'kvert_cos_half', 'kvert_cos_q1', 'kvert_cos_q3']

X = df[feature_cols]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================================================================
# UNDERSAMPLING: уравниваем количество IS и GS
# ===================================================================
print(f"\nДо балансировки: IS={np.sum(y_train==0)}, GS={np.sum(y_train==1)}")

# Разделяем на IS и GS
X_train_is = X_train[y_train == 0]
X_train_gs = X_train[y_train == 1]
y_train_is = y_train[y_train == 0]
y_train_gs = y_train[y_train == 1]

# Оставляем все IS, но берём случайную подвыборку GS такого же размера
n_is = len(X_train_is)
X_train_gs_down = resample(X_train_gs, n_samples=n_is, random_state=42, replace=False)
y_train_gs_down = np.zeros(n_is, dtype=int) + 1  # все 1 (GS)

# Объединяем обратно
X_train_balanced = pd.concat([X_train_is, X_train_gs_down])
y_train_balanced = pd.concat([y_train_is, pd.Series(y_train_gs_down)])

# Перемешиваем
shuffle_idx = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced.iloc[shuffle_idx]
y_train_balanced = y_train_balanced.iloc[shuffle_idx]

print(f"После балансировки: IS={np.sum(y_train_balanced==0)}, GS={np.sum(y_train_balanced==1)}")

# ===================================================================
# Обучаем XGBoost на сбалансированных данных
# ===================================================================
print("\nОбучаю XGBoost на сбалансированных данных...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)

# ===================================================================
# Подбираем порог
# ===================================================================
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\nЗависимость метрик от порога:")
print("Threshold | IS recall | GS recall | F1")
print("-" * 45)

best_is_recall = 0
best_threshold = 0.5

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    report = classification_report(y_test, y_pred_t, output_dict=True)
    is_recall = report['0']['recall']
    gs_recall = report['1']['recall']
    print(f"    {t:.1f}    |   {is_recall:.4f}  |   {gs_recall:.4f}  | {f1:.4f}")
    
    if is_recall > best_is_recall:
        best_is_recall = is_recall
        best_threshold = t

# ===================================================================
# Финальный результат
# ===================================================================
y_pred_final = (y_proba >= best_threshold).astype(int)

print(f"\n{'='*50}")
print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (порог={best_threshold})")
print(f"{'='*50}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_final):.4f}")
print(classification_report(y_test, y_pred_final, target_names=['IS (0)', 'GS (1)']))