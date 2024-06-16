# Определение дефектов сварных швов с помощью YOLOv9-E

Ниже будет представлена подробная инструкция по работе с моделью YOLOv9-E для определения дефектов сварных швов на фото


# Подготовка данных

Прежде всего необходимо загрузить предоставленные данные в корневую папку проекта.
Также необходимо загрузить модель и все ее зависимости.

    !git clone https://github.com/WongKinYiu/yolov9.git
    !wget -P weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt  
    %cd yolov9
    %pip install -r requirements.txt

## Аугментация данных

Далее, для того чтобы модель не зацикливалась на конкретных признаках необходимо проаугментировать данные. Для этого необходимо запустить следующий код:

    import albumentations as A  
    import cv2  
    import os  
    import torch  
    import numpy as np  
    from albumentations.pytorch import ToTensorV2  
       
    transform = A.Compose([  
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),  
        A.GaussianBlur(blur_limit=(3, 7), p=0.7),  
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),  
        A.RandomGamma(gamma_limit=(80, 120), p=0.7),  
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.7),  
        A.ToGray(p=0.7),  
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),  
        ToTensorV2()  
    ])
    
    ...

## Формирование датасета

Теперь следует разделить полученные данные на тренировочную и валидационную выборки.
Для этого необходимо выполнить следующий код:

    import os  
    import shutil  
    import random  
      
    base_dir = 'velding_defects' # Замените на ваш фактический путь к датасету  
    images_dir = os.path.join(base_dir, 'images')  
    labels_dir = os.path.join(base_dir, 'labels')  
       
    image_train_dir = os.path.join(images_dir, 'train')  
    image_val_dir = os.path.join(images_dir, 'val')  
    label_train_dir = os.path.join(labels_dir, 'train')  
    label_val_dir = os.path.join(labels_dir, 'val')
    
    ...

Теперь нужно создать файлы train.txt и val.txt, где будут храниться пути до изображений. Для этого надо выполнить следующий код:

    import os  
    from pathlib import Path  
      
    base_dir = Path('velding_defects')  
    train_images_dir = base_dir / 'images' / 'train'  
    val_images_dir = base_dir / 'images' / 'val'  
      
    train_txt_path = base_dir / 'train.txt'  
    val_txt_path = base_dir / 'val.txt'
    
    ...
Последним шагом будет создание data.yaml файла с конфигурацией датасета, который должен выглядеть следующим образом:

    train: velding_defects/train.txt
    val: velding_defects/val.txt
    nc: 5
    names: ['adj', 'int', 'geo', 'pro', 'non']
 По итогу должна получиться вот такая структура:
 

    velding_defects/
    │
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── test/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    │
    ├── labels/
    │   ├── train/
    │   │   ├── image1.txt
    │   │   ├── image2.txt
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.txt
    │   │   ├── image2.txt
    │   │   └── ...
    │   └── test/
    │       ├── image1.txt
    │       ├── image2.txt
    │       └── ...
    │
    ├── train.txt
    ├── val.txt
    └──test.txt

# Обучение

Теперь необходимо перенести сформированный датасет в папку со скачанной YOLOv9. Чтобы запустить обучение необходимо выполнить следующую команду:

    %cd yolov9
    !python3 -m torch.distributed.run --nproc_per_node 2 train_dual.py \  
    --batch 32 --epochs 30 --img 640 --device 0,1 --min-items 0 --close-mosaic 5 \  
    --data velding_defects/data.yaml \  
    --weights /home/jupyter/work/resources/weights/yolov9-e.pt \  
    --cfg models/detect/yolov9-e.yaml \  
    --hyp hyp.scratch-high.yaml
Когда обучение завершится веса, графики и метрики будут находиться по пути: **yolov9/runs/train/exp<#>/**
# Валидация

Для проверки переобучения модели может понадобиться провести валидацию:

    !python3 val_dual.py \  
    --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 \  
    --data velding_defects/data.yaml \  
    --weights runs/train/exp14/weights/best.pt\
 Когда валидация завершится графики и метрики будут находиться по пути: **yolov9/runs/val/exp<#>/**

# Детекция

Теперь когда мы уверены в работоспособности модели, применим ее на практике. Для этого понадобится выполнить следующий код:

    from yolov9.detect_dual import run  
      
    run(  
        weights='/home/jupyter/work/resources/yolov9/runs/train/exp14/weights/best.pt',  
        conf_thres=0.1,  
        imgsz=(640,640),  
        source='/home/jupyter/work/resources/*изображения для детекции*',  
        save_txt=True,  
        nosave=True, #Опционально  
        device='0'  
    )
Полученные файлы разметки и изображения с выделенными областями находятся по пути: **yolov9/runs/detect/exp<#>/**
