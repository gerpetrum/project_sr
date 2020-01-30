# Super Resolution
## 1. Setting environment
Клонирование репозиотрия проекта:
```shell script
git clone https://sbtatlas.sigma.sbrf.ru/stash/scm/dsidp/super-resolution.git
```
Обучающая выбока генерируется синтетически их pdf-файлов взятых из репозиториев: [pdfs](https://github.com/tpn/pdfs) и [rupdfs](https://github.com/gerpetrum/rupdfs)

Генерирование данных для обучения:
```shell script
cd super-resolution
chmod +x ./script.sh
./script.sh
```
``script_generate_synthetic.py`` - утилита для генерирования своего синтетического набора из PDF-файлов.
```shell script
usage: script_generate_synthetic.py [-h] [-s SOURCE_DIRECTORY]
                                    [-d DESTINATION_DIRECTORY] [-r RESOLUTION]
                                    [-t THREADS] [-l LOGGING] [-n NAME]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE_DIRECTORY, --source_directory SOURCE_DIRECTORY
                        super resolution upscale factor
  -d DESTINATION_DIRECTORY, --destination_directory DESTINATION_DIRECTORY
                        testing batch size
  -r RESOLUTION, --resolution RESOLUTION
                        dpi image
  -t THREADS, --threads THREADS
                        number of threads for Samples synthetic to use
  -l LOGGING, --logging LOGGING
                        directory for log-file
  -n NAME, --name NAME  name dataset

```

Пример запуска генератора синтетики:
```shell script
python3 ./script_generate_synthetic.py -s ../pdfs -r 300 -d ../Samples -t 4 -n pdfs_300_dpi
```

<p align="center">
  <img width="800" src="Figures/crops.jpg">
  <br> Примеры сгенерированных изображений (dpi 300, 150, 75, 60, 50)
</p>

Установка требуемых библиотек:
```shell script
pip3 install -r requirements.txt
```

- ``check_error_channels_on_graund_truth.ipynb`` - тетрадка для поканального сравнительного анализа исходного изображения в высоком качестве с сгенерированным изображением по образцу из низкого разрешения.
- ``example_predict.ipynb`` - тетерадка с примером работы (predict) модели.
- ``run_train_rdn.py`` - скрипт запуска обучения модели RDN.
- ``run_train_rrdn.py`` - скрипт запуска обучения модели RRDN.
- ``script_generate_synthetic.py`` - утилита для генерирования синтетического набора из PDF-файлов.
- ``ISR`` - библиотека для обучения и предсказания моделей RRDN/RDN.
- ``Results`` - результаты проведенных экспериментов.
- ``Samples`` - примеры синтетических данных с разным dpi, примеры для валидации модели ``Samples/analyse``.
- ``logs`` - логи экспериментов (системный лог + tensorboard лог).
- ``weights`` - веса обученных моделей и конфигурации процесса обучения.

## 2. Train
Обучение моделей:
```shell script
echo "Run train RRDN model"
python3 run_train_rrdn.py
```

```shell script
echo "Run train RDN model"
python3 run_train_rdn.py
```

## 3. Predict
Пример запуска модели для повышения качества изображения. Тетрадка с выполнением модеи ``example_predict.ipynb`` содержит пример использования модели и визуализацию с исходным изображением и интерполяцией.
```python
from ISR.models import RDN, RRDN
from PIL import Image
import numpy as np

weights = 'weights/pre-train/rrdn-C4-D3-G32-G032-T10-x4/Perceptual/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5'
C = 4
D = 3
G = 32
G0 = 32
T = 10
scale = 4
lr = 'Samples/lr.jpg'
sr = 'Samples/hr.jpg'
dpi = 300

if type is 'RDN':
    rdn = RDN(arch_params={'C': C, 'D': D, 'G': G, 'G0': G0, 'T': T, 'x': scale})
else:
    rdn = RRDN(arch_params={'C': C, 'D': D, 'G': G, 'G0': G0, 'T': T, 'x': scale})

rdn.model.load_weights(weights)
image_sr = rdn.predict(np.array(Image.open(lr)))
Image.fromarray(image_sr).save(sr, dpi=(dpi, dpi))
```

# 4. Monitoring
Настроено логирование функций ошибок и метрик на обучении и валидации.
```shell script
tensorboard --logdir ./logs --host 2020-01-04_22_04_54 --port 9090
```

