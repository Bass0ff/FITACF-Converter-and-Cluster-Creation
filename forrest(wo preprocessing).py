import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

pd.options.mode.chained_assignment = None

print("Загружаю датасет...")
df = pd.read_csv("dataset_for_ml_labeled.csv")
print(f"Загружено {len(df)} записей")

feature_cols = ['vel', 'width', 'height_out', 'hop_out', 'kvert_cos', 'kb_cos', 
                'kvert_cos_half', 'kvert_cos_q1', 'kvert_cos_q3']

X = df[feature_cols]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nОбучаю оптимизированный Random Forest...")
rf = RandomForestClassifier(
    n_estimators=150,           # Больше деревьев
    max_depth=15,               # Глубже
    class_weight={0: 3, 1: 1},    # Борьба с дисбалансом
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\n" + "="*50)
print("РЕЗУЛЬТАТ ОПТИМИЗИРОВАННОГО RANDOM FOREST")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['IS (0)', 'GS (1)']))