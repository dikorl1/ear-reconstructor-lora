# ear-reconstructor-lora

🎨 **AI-powered ear reconstruction for portrait enhancement. Two-stage LoRA training (DreamBooth + Inpaint) on Stable Diffusion 1.5. Age-aware, pose-aware, anatomically correct. Photorealistic results with automatic quality control.**

---

## 📋 О проекте
Система обучения LoRA-моделей и инпейнтинга для фотореалистичной реконструкции уха на портретах. Поддерживает несколько возрастных групп, разные ракурсы (front/side/back), позы (standing/sitting/supine/prone), а также автоматический QC датасета и балансировку выборки.

---

🔧 Что умеет
[](https://github.com/USER/ear-reconstructor-lora-inpaint#-%D1%87%D1%82%D0%BE-%D1%83%D0%BC%D0%B5%D0%B5%D1%82)
– Обучать Stage A DreamBooth‑LoRA (prior‑preservation) на полнотелесных портретах и крупных планах ушей  
– Обучать Stage B Inpaint‑LoRA на парах (image + mask + masked image) для точного восстановления уха  
– Строить датасет по age buckets (15–17, 18–22, 23–25, 26–30, 31–35) и разным позам/ракурсам  
– Генерировать txt2img full‑body/torso/bust портреты с анатомически корректным ухом  
– Делать инпейнт уха по маске ( SAM/ручная или синтетически сгенерированная)  
– Автоматически проверять качество (blur, JPEG blockiness, noise, размер/кадрирование) и отбраковывать неподходящие кадры  
– Поддерживать ControlNet (OpenPose/Depth/Normal) и IP‑Adapter для более стабильной геометрии и внешнего вида  

🛠 Используемые технологии
[](https://github.com/USER/ear-reconstructor-lora-inpaint#-%D0%B8%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D1%83%D0%B5%D0%BC%D1%8B%D0%B5-%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B8)
– PyTorch  
– HuggingFace Diffusers (StableDiffusionPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler)  
– LoRA (Low-Rank Adaptation) + DreamBooth  
– ControlNet (OpenPose/Depth/Normal), IP‑Adapter  
– PIL, OpenCV, NumPy, SciPy для обработки изображений и QC  
– Accelerate, xFormers, 8‑bit Adam, gradient checkpointing для обучения на GPU 16 GB  
– JSON/JSONL‑манифесты для описания датасета (age bucket, pose, view, type, earvisible, prompt/negative)  

📊 Результаты проекта
[](https://github.com/USER/ear-reconstructor-lora-inpaint#-%D1%80%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%D1%8B-%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0)
🎯 Фотореалистичная реконструкция уха с учётом:
– возрастного диапазона (15–35 лет)  
– формы уха (slightly protruding / flat-attached / average)  
– типа мочки (attached / detached / small)  
– тона кожи (light beige / warm beige / olive / medium tan / fair / deep warm)  

📉 Снижение количества артефактов  
– Убраны типичные проблемы: «plastic / waxy skin», «oversized ear», extra ears, неправильная геометрия  
– Negative‑промпты и prior‑preservation уменьшают mode collapse и деградацию языка  

⚡ Производительность пайплайна  
– Stage A (DreamBooth‑LoRA): 1200–1800 шагов, LR ~ 5e‑5, rank 16–32  
– Stage B (Inpaint‑LoRA): 600–1200 шагов, LR 5e‑5–1e‑4, mask‑aware U‑Net  
– Инференс: 40 шагов txt2img + 36–44 шагов inpaint, CFG 4.5–5.5  

🧠 Умный sampler и QC  
– Автобалансировка по классам: clients.back / clients.torso_full / clients.portrait / clients.full  
– Автоматика по VarLaplacian, blockiness, шуму и размеру кадра  
– Отдельные buckets для front / profile left / profile right / back, standing / sitting / supine / prone, ear left/right/both  

🚀 Быстрый старт
[](https://github.com/USER/ear-reconstructor-lora-inpaint#-%D0%B1%D1%8B%D1%81%D1%82%D1%80%D1%8B%D0%B9-%D1%81%D1%82%D0%B0%D1%80%D1%82)
1. Установить зависимости:
   pip install -r requirements.txt

2. Подготовить данные по структуре из docs/DATA_PREPARATION.md  
3. Запустить Stage A (DreamBooth‑LoRA)  
4. Запустить Stage B (Inpaint‑LoRA) с использованием LoRA из Stage A  
5. Выполнить генерацию/инпейнт по примеру из docs/PROMPTS.md  

⚠️ О проекте
[](https://github.com/USER/ear-reconstructor-lora-inpaint#%EF%B8%8F-%D0%BE-%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B5)
📁 Репозиторий содержит демонстрационную структуру кода и примеры конфигураций для обучения LoRA/Inpaint‑моделей под задачу реконструкции уха.

– Полный продакшн‑датасет не включён и описывается через manifest‑файлы (JSON/JSONL)  
– Часть параметров (бренды моделей, пути к датасету, конфигурации обучения) могут быть изменены под конкретный проект/GPU  
– Подробные манифесты и примеры запуска с реальными данными могут быть предоставлены по запросу с NDA  

📫 Контакты
[](https://github.com/USER/ear-reconstructor-lora-inpaint#-%D0%BA%D0%BE%D0%BD%D1%82%D0%B0%D0%BA%D1%82%D1%8B)
Telegram: https://t.me/workdmitrii  
Email: korlyakov.dmitry.n@yandex.ru
