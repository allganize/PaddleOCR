import os
import subprocess

text_detection_model = "../ppstructure/inference/ch_PP-OCRv3_det_infer"
# text_recognition_char_dict = "../ppocr/utils/ppocr_keys_v1.txt"
# text_recognition_model = "inference/ch_PP-OCRv3_rec_infer"
text_recognition_char_dict = "../ppocr/utils/dict/japan_dict.txt"
text_recognition_model = "../ppstructure/inference/japan_PP-OCRv3_rec_infer"

table_char_dict = "../ppocr/utils/dict/table_structure_dict_ch.txt"
table_recognition_model = "Baseline"


for file_name in os.listdir("../data/input/"):
    table_detect_result = subprocess.getstatusoutput(f"python table_detect.py {file_name}")[1]

    recognition_result = subprocess.getstatusoutput(f"python ../ppstructure/table/predict_table.py \
        --det_model_dir={text_detection_model} \
        --rec_model_dir={text_recognition_model} \
        --table_model_dir=../model/{table_recognition_model} \
        --rec_char_dict_path={text_recognition_char_dict} \
        --table_char_dict_path={table_char_dict} \
        --image_dir=../data/output/table_detection/{file_name} \
        --output=../data/output/{table_recognition_model}")
    
    print(file_name)
    