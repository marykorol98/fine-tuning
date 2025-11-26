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
import logging
import os
import random
from pathlib import Path

from model.lama_wrapper import ModelLamaLLM
logger = logging.getLogger(__name__)


# ---------------- Настройки ----------------
TABLES_PATH = "data/schemas/table_schemas.json"  # где лежат schema+preview
MODEL_NAME = "qwen2.5:14b"

OUTPUT_FILE = f"data/instructions/sft_dataset_{MODEL_NAME}.jsonl"
BATCH_SIZE = 300
NUM_SAMPLES = 3000

random.seed(42)

# ---------------- Шаблоны инструкций ----------------
# instruction_templates = {
#     "EDA": [
#         "Опиши распределение {column}, среднее, медиана, выбросы.",
#         "Сделай summary таблицы {table}, включая базовые статистики и кол-во пропусков."
#     ],
#     "Diagnostics": [
#         "Найди возможные проблемы в таблице {table}: nulls, duplicates, некорректные значения.",
#         "Проанализируй колонку {column} на потенциальные ошибки и пропуски."
#     ],
#     "Plots": [
#         "Построй plotly график, наглядно показывающий распределение колонки {column}.",
#         "Создай интерактивный график для {column} с использованием plotly."
#     ],
#     "Recommendation": [
#         "Как бы ты предложил поработать с таблицей {table} для улучшения данных?",
#         "Дай рекомендации по предобработке таблицы {table} перед ML задачами."
#     ]
# }

TYPES = [
    "EDA: Опиши распределение price, среднее, медиана, выбросы.",
    "Diagnostics: Найди возможные проблемы в данных (nulls, duplicates).",
    "Plots: Построй plotly график, наиболее наглядно показывающий распределение данных.",
    "Recommendation: Что бы ты предложил улучшить в данных?"
]

PROMPT_TEMPLATE = """
Ты — ассистент, который генерирует SFT-инструкции для обучения модели.
Каждая инструкция должна быть уникальной, содержательной и связанной с анализом данных.

Сгенерируй {n} уникальных инструкций.
Используй следующий стиль:

Типы:
- EDA
- Diagnostics
- Plots
- Recommendation

Формат вывода JSON:
[
  {{
    "instruction": "...",
    "type": "EDA | Diagnostics | Plots | Recommendation"
  }},
  ...
]
"""


# ---------------- Основная генерация ----------------
def main():
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    written = 0
    batch_id = 1
        
    llm_model = ModelLamaLLM(MODEL_NAME) 
    prompt = PROMPT_TEMPLATE.format(n=BATCH_SIZE)
    
    while written < NUM_SAMPLES:
        count = min(BATCH_SIZE, NUM_SAMPLES - written)
        logger.info(f"→ Генерация batch {batch_id} ({count} примеров)")

        batch = llm_model.generate_output(prompt)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        written += len(batch)
        batch_id += 1

        logger.info(f"✓ Готово: {written}/{NUM_SAMPLES}")

    logger.info("\nFINISHED! Датасет сохранён в", OUTPUT_FILE)
    

    # for batch_start in range(0, NUM_SAMPLES, BATCH_SIZE):
    #     batch_end = min(batch_start + BATCH_SIZE, NUM_SAMPLES)
    #     logger.info(f"Генерирую batch {batch_start}–{batch_end}...")
    #     prompt = PROMPT_TEMPLATE.format(n=BATCH_SIZE)

    #     dataset = []

    #     for _ in tqdm(range(batch_start, batch_end), desc="Generating SFT samples"):
    #         table_entry = random.choice(tables)
    #         table_name = table_entry["file"].replace(".csv","")
    #         columns = list(table_entry["schema"].keys())

    #         instr_type = random.choice(list(instruction_templates.keys()))
    #         template = random.choice(instruction_templates[instr_type])

    #         column = random.choice(columns) if "{column}" in template else None
    #         instruction = template.format(table=table_name, column=column)

    #         context = {
    #             "schema": table_entry["schema"],
    #             "preview": table_entry["preview"]
    #         }

    #         output = llm_model.generate_output(instruction, context)

    #         record = {
    #             "id": str(uuid.uuid4()),
    #             "instruction": instruction,
    #             "context": context,
    #             "output": output,
    #             "meta": {
    #                 "table": table_name,
    #                 "columns": columns,
    #                 "type": instr_type,
    #                 "auto_generated": True
    #             }
    #         }

    #         dataset.append(record)

    #         # Опционально: записывать по частям, чтобы не потерять прогресс
    #         if len(dataset) % 50 == 0:
    #             with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    #                 for rec in dataset:
    #                     f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    #             dataset = []

    #     # Запись оставшихся
    #     if dataset:
    #         with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    #             for rec in dataset:
    #                 f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
    #     logger.info(f"✔ Batch {batch_start}–{batch_end} готов")

    # logger.info(f"Генерация завершена. Примеры сохранены в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
