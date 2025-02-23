import functools
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from modelscope.msdatasets import MsDataset

from swanlab.integration.huggingface import SwanLabCallback

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from multiprocessing import freeze_support  # 导入freeze_support

def dataset_jsonl_transfer(origin_dataset, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    for line in origin_dataset:
        # 解析每一行的json数据
        data = line
        context = data["text"]
        catagory = data["category"]
        label = data["output"]
        message = {
            "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
            "input": f"文本:{context},类型选型:{catagory}",
            "output": label,
        }
        messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
            
            
def process_func(tokenizer, example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(device, messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    #model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response
    

if __name__ == '__main__':
    freeze_support()  # 如果程序不会被冻结成可执行文件，这行可以省略
    # 在modelscope上下载Qwen模型到本地目录下
    if not os.path.exists("./qwen/Qwen2-1___5B-Instruct/"):
        snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")

    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 检查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"this device is {device}")
    # 确保模型在相应的设备上
    model.to(device)

    # 加载、处理数据集和测试集
    train_dataset = MsDataset.load('swift/zh_cls_fudan-news', split='train', trust_remote_code=True)[:16]
    test_dataset = MsDataset.load('swift/zh_cls_fudan-news', subset_name='test', split='test')

    train_jsonl_new_path = "new_train.jsonl"
    test_jsonl_new_path = "new_test.jsonl"

    if not os.path.exists(train_jsonl_new_path):
        dataset_jsonl_transfer(train_dataset, train_jsonl_new_path)
    if not os.path.exists(test_jsonl_new_path):
        dataset_jsonl_transfer(test_dataset, test_jsonl_new_path)

    # 得到训练集
    train_df = pd.read_json(train_jsonl_new_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)

     # 使用 functools.partial 固定 tokenizer 参数
    process_func_with_tokenizer = functools.partial(process_func, tokenizer)
    train_dataset = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names,num_proc=32)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )

    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir="./output/Qwen1.5",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=100,
        learning_rate=1e-2,
        save_on_each_node=True,
        gradient_checkpointing=False,
        report_to="none",
    )

    swanlab_callback = SwanLabCallback(
        project="train-qwen2-category",
        workspace="ai-next-furture",
        experiment_name="Qwen2-1.5B-Instruct",
        description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
        config={
            "model": "qwen/Qwen2-1.5B-Instruct",
            "dataset": "huangjintao/zh_cls_fudan-news",
        }
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    trainer.train()

    # 用测试集的前10条，测试模型
    test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]

    test_text_list = []
    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']
        
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(device, messages, model, tokenizer)
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
        test_text_list.append(swanlab.Text(result_text, caption=response))
        
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()
