import sys

import torch
from PIL import Image, ImageDraw
from transformers import DetrFeatureExtractor, DetrForObjectDetection

image_dir = "../data/input/"
output_dir = "../data/output/table_detection/"
detect_model_dir = "TahaDouaji/detr-doc-table-detection"
image_name = sys.argv[1]

image = Image.open(image_dir + image_name)

feature_extractor = DetrFeatureExtractor.from_pretrained(detect_model_dir)
model = DetrForObjectDetection.from_pretrained(detect_model_dir)

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

score_list = list(results["scores"])
max_val = max(score_list)
max_id = score_list.index(max_val)

x_min, y_min, x_max, y_max = list(results["boxes"][max_id])
x_min, y_min, x_max, y_max = x_min.item(), y_min.item(), x_max.item(), y_max.item()

padding = 50
croppedImage = image.crop(
    (x_min - padding, y_min - padding, x_max + padding, y_max + padding)
)
croppedImage.save(output_dir + image_name)
print("Finish detection of table")
