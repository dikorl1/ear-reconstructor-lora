Краткое содержание:

docs/README.md
Обзор документации по разделам.

STAGE_A_TRAINING.md
– Команда accelerate launch ... с параметрами из 11.txt и Trebovaniia‑k‑kodu…
– Таблица ключевых параметров: LR, prior_loss_weight, rank, batch size, steps.
– Рекомендации по age buckets и распределению промптов.

STAGE_B_TRAINING.md
– Описание Stage B (Inpaint LoRA):

загрузка Stage A LoRA,

использование image/mask/masked,

параметры inpaint (denoise, CFG, steps),

интеграция ControlNet/Refiner при необходимости.

DATA_PREPARATION.md
– Все, что у тебя подробно описано: разрешения, кропы, occluded, clients, front/side/back, earscloseup, JSONL‑манифесты.
– Пороговые значения VarLaplacian, blockiness и т.д.

QC_PIPELINE.md
– Алгоритм QC из Trebovaniia‑k‑kodu:

VarLaplacian пороги,

min side 512/768,

ear‑box ≥ 200 px,

классы статуса OK/WARN/REJECT,

структура qcreport.csv.

PROMPTS.md
– Собраны лучшие примеры промптов и negative‑промптов:

для full‑body/torso/bust,

для ear closeup,

для occluded (hair/bandage/accessories),

примеры age‑specific промптов (15–17, 18–22, 23–25, 26–30, 31–35).

