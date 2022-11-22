import subprocess

pretrained_model_config = "../model/pretrain_models/ch_ppstructure_mobile_v2.0_SLANet_train"
character_dict_path = "../ppocr/utils/dict/table_structure_dict_ch.txt"

data_name = "PubTabDataSet"
data_dir = "../data/train_data/simple_table"
train_data_label = ["../data/train_data/simple_table/train.txt"]
val_data_label = ["../data/train_data/simple_table/val.txt"]

finetune_model_dir = "Exp2_finetune"
export_model_dir = "Exp2_inference"

finetune_cmd = f"python ../tools/train.py -c {pretrained_model_config}/config.yml -o \
Global.epoch_num=50 \
Global.print_batch_step=5 \
Global.save_model_dir=../model/{finetune_model_dir} \
Global.pretrained_model={pretrained_model_config}/best_accuracy.pdparams \
Global.eval_batch_step=[0,375] \
Global.character_dict_path={character_dict_path} \
Optimizer.lr.name=Const \
Optimizer.lr.learning_rate=0.0005 \
Train.dataset.name={data_name} \
Train.dataset.data_dir={data_dir} \
Train.dataset.label_file_list={train_data_label} \
Eval.dataset.name={data_name} \
Eval.dataset.data_dir={data_dir} \
Eval.dataset.label_file_list={val_data_label}"

process = subprocess.Popen(finetune_cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

export_cmd  = f"python ../tools/export_model.py \
    -c ../model/{finetune_model_dir}/config.yml \
    -o Global.pretrained_model=../model/{finetune_model_dir}/latest \
    Global.save_inference_dir=../model/{export_model_dir}/"

export_result = subprocess.getstatusoutput(export_cmd)[1]
print(export_result)