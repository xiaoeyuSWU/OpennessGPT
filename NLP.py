import torch
import os
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 定义一个自定义数据集类，用于处理内存中的文本数据
class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size):
        self.tokenizer = tokenizer
        # 在分词时调整填充到最大长度
        self.inputs = [tokenizer(text, truncation=True, max_length=block_size, padding="max_length", return_tensors="pt") for text in texts]

    def __len__(self):
        # 返回数据集的大小
        return len(self.inputs)

    def __getitem__(self, idx):
        # 返回指定索引的数据项，包括输入id和注意力掩码
        return {"input_ids": self.inputs[idx]["input_ids"].squeeze(), "attention_mask": self.inputs[idx]["attention_mask"].squeeze()}

# 从包含文本文件的目录加载数据的函数
def load_dataset(directory_path):
    texts = []
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 只处理以 .txt 结尾的文件
        if filename.endswith(".txt"):
            # 打开并读取文件内容，去除换行符后添加到文本列表中
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read().replace('\n', ' '))
    return texts

# 训练模型的函数
def train_model(texts, model_name, output_dir):
    print("加载分词器和模型...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("分词器和模型加载完成。")

    tokenizer.pad_token = tokenizer.eos_token  # 确保填充标记设置正确

    print("创建数据集...")
    dataset = CustomTextDataset(tokenizer, texts, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("设置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,  # 模型输出目录
        overwrite_output_dir=True,  # 是否覆盖输出目录
        num_train_epochs=3,  # 训练的轮数
        per_device_train_batch_size=2,  # 每个设备上的训练批次大小
        save_steps=100,  # 每隔多少步保存一次模型
        save_total_limit=2,  # 保存模型的总数量限制
        prediction_loss_only=True,  # 仅计算预测损失
    )

    print("初始化训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    print("开始训练...")
    trainer.train()  # 开始训练
    print("训练完成。")

    print("保存模型...")
    model.save_pretrained(output_dir)  # 保存模型
    tokenizer.save_pretrained(output_dir)  # 保存分词器
    print("模型保存完成。")

# 从训练好的模型生成文本的函数
def generate_text(model_path, start_prompt="Today is a good day so", max_length=100):
    # 加载训练好的模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 准备输入，包含适当的注意力掩码和标记类型
    encoded_inputs = tokenizer(start_prompt, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    # 调整生成的最大长度
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length + 50,  # 允许模型生成内容的空间
        temperature=1.0,  # 控制生成文本的随机性
        no_repeat_ngram_size=2  # 防止重复生成相同的单词/短语
    )
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# 主执行逻辑：加载数据，训练模型并生成文本
if __name__ == "__main__":
    try:
        print("加载数据...")
        high_openness_texts = load_dataset('high')  # 加载高开放性数据集
        low_openness_texts = load_dataset('low')  # 加载低开放性数据集
        print("数据加载成功。")

        print("训练高开放性模型...")
        train_model(high_openness_texts, 'gpt2', 'high_openness_model')  # 训练高开放性模型
        print("高开放性模型训练完成。")

        print("训练低开放性模型...")
        train_model(low_openness_texts, 'gpt2', 'low_openness_model')  # 训练低开放性模型
        print("低开放性模型训练完成。")

        print("从高开放性模型生成文本...")
        print("高开放性示例:", generate_text('high_openness_model'))  # 从高开放性模型生成示例文本

        print("从低开放性模型生成文本...")
        print("低开放性示例:", generate_text('low_openness_model'))  # 从低开放性模型生成示例文本
    except Exception as e:
        print("发生错误:", str(e))  # 捕获并打印错误信息
