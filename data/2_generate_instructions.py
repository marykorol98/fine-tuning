instruction_templates = {
    "EDA": [
        "Опиши распределение {column}, среднее, медиана, выбросы.",
        "Сделай summary таблицы {table}, включая базовые статистики и кол-во пропусков."
    ],
    "Diagnostics": [
        "Найди возможные проблемы в таблице {table}: nulls, duplicates, некорректные значения.",
        "Проанализируй колонку {column} на потенциальные ошибки и пропуски."
    ],
    "Plots": [
        "Построй plotly график, наглядно показывающий распределение колонки {column}.",
        "Создай интерактивный график для {column} с использованием plotly."
    ],
    "Recommendation": [
        "Как бы ты предложил поработать с таблицей {table} для улучшения данных?",
        "Дай рекомендации по предобработке таблицы {table} перед ML задачами."
    ]
}


import json
import random
import uuid
from pathlib import Path
from model.gpt_4 import ModelGPT

# ---------------- Настройки ----------------
TABLES_PATH = "data/schemas/table_schemas.json"  # где лежат schema+preview
OUTPUT_FILE = "data/instructions/sft_dataset_gpt_4.jsonl"
NUM_SAMPLES = 3000
random.seed(42)

# ---------------- Шаблоны инструкций ----------------
instruction_templates = {
    "EDA": [
        "Опиши распределение {column}, среднее, медиана, выбросы.",
        "Сделай summary таблицы {table}, включая базовые статистики и кол-во пропусков."
    ],
    "Diagnostics": [
        "Найди возможные проблемы в таблице {table}: nulls, duplicates, некорректные значения.",
        "Проанализируй колонку {column} на потенциальные ошибки и пропуски."
    ],
    "Plots": [
        "Построй plotly график, наглядно показывающий распределение колонки {column}.",
        "Создай интерактивный график для {column} с использованием plotly."
    ],
    "Recommendation": [
        "Как бы ты предложил поработать с таблицей {table} для улучшения данных?",
        "Дай рекомендации по предобработке таблицы {table} перед ML задачами."
    ]
}

# ---------------- Основная генерация ----------------
def main():
    # Создаём output директорию
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Загружаем таблицы
    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        tables = json.load(f)

    dataset = []

    for _ in range(NUM_SAMPLES):
        table_entry = random.choice(tables)
        table_name = table_entry["file"].replace(".csv","")
        columns = list(table_entry["schema"].keys())

        # Случайный тип инструкции
        instr_type = random.choice(list(instruction_templates.keys()))
        template = random.choice(instruction_templates[instr_type])

        # Подставляем table/column
        column = random.choice(columns) if "{column}" in template else None
        instruction = template.format(table=table_name, column=column)

        llm_model = ModelGPT()
        # Генерация output
        output = llm_model.generate_output(instr_type, table_name, column)

        # Формируем запись
        record = {
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "context": {
                "schema": table_entry["schema"],
                "preview": table_entry["preview"]
            },
            "output": output,
            "meta": {
                "table": table_name,
                "columns": columns,
                "type": instr_type,
                "auto_generated": True
            }
        }

        dataset.append(record)

    # Сохраняем в JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Генерация завершена. {NUM_SAMPLES} примеров сохранены в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
