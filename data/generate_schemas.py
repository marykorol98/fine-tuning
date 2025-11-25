import os
import json
import pandas as pd
from typing import Dict, Any

# -------- Настройки --------
INPUT_DIR = "data/tables"     # Папка с твоими CSV
OUTPUT_DIR = "data/schemas" # Куда сохранить JSON-датасет
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Функция определения типа --------
def detect_type(value):
    """
    Примитивный, но рабочий детектор типов.
    Возвращает строковый тип: int, float, bool, str, null.
    """
    if pd.isna(value):
        return "null"
    
    v = str(value).strip()

    # Bool
    if v.lower() in ["true", "false"]:
        return "bool"

    # Int
    try:
        int(v)
        return "int"
    except:
        pass

    # Float
    try:
        float(v)
        return "float"
    except:
        pass

    # Str
    return "str"


# -------- Генерация schema --------
def generate_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        col_values = df[col].dropna()

        example = col_values.iloc[0] if len(col_values) > 0 else None

        # Преобразуем numpy-типы в стандартные Python-типы
        if example is not None:
            if hasattr(example, "item"):
                example = example.item()  # int64/float64 -> int/float

        inferred_type = detect_type(example)
        schema[col] = {
            "type": inferred_type,
            "example": example,
        }
    return schema


# -------- Генерация preview --------
def generate_preview(df: pd.DataFrame, n: int = 10):
    records = df.head(n).to_dict(orient="records")
    # Конвертируем все numpy-типы внутри preview
    for rec in records:
        for k, v in rec.items():
            if hasattr(v, "item"):
                rec[k] = v.item()
    return records


# -------- Основная обработка --------
dataset = []

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)

    entry = {
        "file": filename,
        "schema": generate_schema(df),
    }
    
    # Превью 5–10 строк
    entry["preview"] = generate_preview(df, 10)

    dataset.append(entry)

# -------- Сохраняем датасет --------
output_path = os.path.join(OUTPUT_DIR, "table_schemas.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Done! Generated: {output_path}")
