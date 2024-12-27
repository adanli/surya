from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from datetime import datetime

import torch

s0 = datetime.now()
print(torch.__version__)
print(torch.cuda.is_available())

s1 = datetime.now()
image = Image.open('/Users/adan/code/egg/surya/lQLPKGnvh9OHCynNA2fNBmGwydgk7GD-6hcHUiSs38yrAA_1633_871.png')
langs = ["en"] # 替换为你的语言 - 可选但推荐
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

s2 = datetime.now()

predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
s3 = datetime.now()
print(predictions)
print('load model time:', s1 - s0)
print('load model time:', s2 - s1)
print('load model time:', s3 - s2)

