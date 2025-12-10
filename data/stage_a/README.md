## краткое описание структуры Stage A
Пример data/stage_a/README.md
# Stage A Dataset (DreamBooth-LoRA Prior)

Структура реального датасета (не включена в репозиторий):

content/drive/MyDrive/datasets/
  age1517/
    standing_front/...
    standing_profileleft/...
    standing_profileright/...
    standing_back/...
    sitting_front/...
    ...
    ear_left/...
    ear_right/...
  age1822/
  age2325/
  age2630/
  age3135/

Для обучения Stage A используются JSONL-манифесты из `manifests/`, где для каждого `path` заданы:
- agebucket
- pose
- view
- type (full / torso / bust / ear)
- earside (left / right / both / none)
- earvisible (true / false)
- prompt
- negative
