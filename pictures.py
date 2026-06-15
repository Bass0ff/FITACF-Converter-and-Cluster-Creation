import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# ===================================================================
# ЗАГРУЖАЕМ ДАННЫЕ И ГОТОВУЮ МОДЕЛЬ
# ===================================================================
print("Загружаю датасет...")
df = pd.read_csv("./clustering_superdarn_fork/dataset_for_ml_labeled.csv")

feature_cols = ['vel', 'width', 'height_out', 'hop_out', 'kvert_cos', 'kb_cos', 
                'kvert_cos_half', 'kvert_cos_q1', 'kvert_cos_q3']

# Загружаем уже обученную модель
print("Загружаю модель из файла...")
model = xgb.XGBClassifier()
model.load_model("xgboost_model_for_inference.json")
print("Модель загружена.")

# ===================================================================
# ФУНКЦИЯ ДЛЯ ПОСТРОЕНИЯ ГРАФИКА ДЛЯ ДИАПАЗОНА ДАТ
# ===================================================================
def plot_season_comparison(df, model, feature_cols, dates, season_name, save_path=None):
    # Фильтруем данные за нужные даты
    df_season_list = []
    for i, (y, m, d) in enumerate(dates):
        mask = (df['year'] == y) & (df['month'] == m) & (df['day'] == d)
        df_day = df[mask].copy()
        if len(df_day) > 0:
            # Сдвигаем время: день 1 = 0-24ч, день 2 = 24-48ч, день 3 = 48-72ч
            df_day['time_shifted'] = df_day['time_hours'] + i * 24
            df_season_list.append(df_day)
    
    if not df_season_list:
        print(f"  {season_name}: нет данных за указанные даты")
        return
    
    df_season = pd.concat(df_season_list, ignore_index=True)
    
    # Предсказания модели
    X = df_season[feature_cols]
    df_season['predicted'] = model.predict(X)
    
    # Строим два графика рядом
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Цвета
    colors_true = ['red' if l == 0 else 'blue' for l in df_season['label']]
    colors_pred = ['red' if l == 0 else 'blue' for l in df_season['predicted']]
    
    # Подписи дней для оси X
    n_days = len(dates)
    day_ticks = [i * 24 + 12 for i in range(n_days)]  # середина каждого дня
    day_labels = [f"{d:02d}.{m:02d}" for y, m, d in dates]
    
    for ax, title, colors in [(ax1, f'Рибейро — {season_name}', colors_true),
                               (ax2, f'XGBoost — {season_name}', colors_pred)]:
        ax.scatter(df_season['time_shifted'], df_season['range_km'], 
                    c=colors, s=0.2, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Время (часы)')
        ax.set_ylabel('Дальность (км)')
        ax.set_xlim(0, n_days * 24)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_labels, fontsize=8)
    
    # Общая подпись
    fig.suptitle(f'Сравнение классификации: Рибейро vs XGBoost\n{season_name}', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Сохранено: {save_path}")
    
    plt.show()
    
    # Считаем совпадение
    accuracy = (df_season['label'] == df_season['predicted']).mean()
    print(f"  Совпадение с Рибейро: {accuracy:.2%}")
    print()

# ===================================================================
# СТРОИМ ГРАФИКИ ПО СЕЗОНАМ
# ===================================================================
# Для каждого сезона время сдвинуто, чтобы дни не накладывались:
# День 1: 0-24ч, День 2: 24-48ч, День 3: 48-72ч

seasons = [
    {
        "name": "Зима",
        "dates": [(2021, 1, 1), (2021, 1, 3), (2021, 1, 5)],
        "save": "comparison_winter.png"
    },
    {
        "name": "Весна",
        "dates": [(2021, 4, 1), (2021, 4, 3), (2021, 4, 5)],
        "save": "comparison_spring.png"
    },
    {
        "name": "Лето (сентябрь)",
        "dates": [(2021, 9, 1), (2021, 9, 3), (2021, 9, 5)],
        "save": "comparison_summer.png"
    },
    {
        "name": "Осень",
        "dates": [(2021, 10, 1), (2021, 10, 3), (2021, 10, 5)],
        "save": "comparison_autumn.png"
    },
]

for season in seasons:
    print(f"\nОбрабатываю: {season['name']}...")
    plot_season_comparison(df, model, feature_cols, season["dates"], 
                           season["name"], season["save"])

print("\nГотово! Все графики сохранены.")