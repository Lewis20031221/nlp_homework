import argparse
import logging
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


from torch.utils.data import Dataset

spto = {"Racism": "种族",
        "Region": "地域",
        "Sexism": "性别",
        "LGBTQ": "LGBTQ",
        "others": "其他",
        "non-hate": "不仇恨"}


def read_line_examples_from_file(data_path):
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    return sents, labels


def get_para_asqp_targets(labels):
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            if sp in spto:
                sp = spto[sp]
            if ot == "hate":
                ot = "仇恨"
            else:
                ot = "不仇恨"
            one_quad_sentence = f"{at} | {ac} | {sp} | {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path):
    sents, labels = read_line_examples_from_file(data_path)
    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]
    targets = get_para_asqp_targets(labels)
    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):
        inputs, targets = get_transformed_io(self.data_path)

        for i in range(len(inputs)):
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='chinese-hate-speech', type=str, required=True,
                        help="任务名称")
    parser.add_argument("--dataset", default='hate', type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--model_name_or_path", default='t5-chinese', type=str,
                        help="预训练模型路径或名称")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="使用的 GPU 数量")
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否进行验证")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="是否直接进行验证")
    parser.add_argument("--do_inference", action='store_true',
                        help="是否进行推理")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="训练时的批量大小")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="验证时的批量大小")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="训练轮数")
    parser.add_argument('--seed', type=int, default=42,
                        help="随机种子")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    args = parser.parse_args()

    # 设置输出目录
    output_dir = f"outputs/hate"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(nn.Module):
    def __init__(self, hparams, model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )


def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        lm_labels = batch["target_ids"].to(device)
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'].to(device)
        )

        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_loader)


def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            lm_labels = batch["target_ids"].to(device)
            lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=batch["source_ids"].to(device),
                attention_mask=batch["source_mask"].to(device),
                labels=lm_labels,
                decoder_attention_mask=batch['target_mask'].to(device)
            )

            loss = outputs[0]
            total_loss += loss.item()

    return total_loss / len(val_loader)


def evaluate(data_loader, model, device):
    # 反向映射字典
    spto_reverse = {
        "种族": "Racism",
        "地域": "Region",
        "性别": "Sexism",
        "LGBTQ": "LGBTQ",
        "其他": "others",
        "不仇恨": "non-hate",
        "仇恨": "hate"
    }

    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 模型生成多四元组结果，格式：四元组1 [SEP] 四元组2 [END]
            outs = model.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=128
            )
            dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            outputs.extend(dec)

    # 后处理：直接处理多四元组格式，转换标签并保留 [SEP] 分隔符
    processed_outputs = []
    for output in outputs:
        # 移除 [END] 标记（如果存在）
        output = output.replace(" [END]", "").strip()

        # 按 [SEP] 分割四元组
        quads = output.split(" [SEP] ")

        # 处理每个四元组，转换后两个元素的标签
        converted_quads = []
        for quad in quads:
            elements = quad.strip().split(" | ")
            if len(elements) == 4:
                at, ac, sp, ot = elements
                sp = spto_reverse.get(sp, sp)
                ot = spto_reverse.get(ot, ot)
                converted_quads.append(f"{at} | {ac} | {sp} | {ot}")
            else:
                converted_quads.append(quad)

        processed_output = " [SEP] ".join(converted_quads) + " [END]"
        processed_outputs.append(processed_output)

    save_results(processed_outputs, "final_results.txt")
    return processed_outputs


def save_results(outputs, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(f"{output}\n")
    print(f"推理结果已保存到 {file_path}")


class LoggingCallback:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, val_loss, epoch):
        self.logger.info(f"***** Validation results after epoch {epoch} *****")
        self.logger.info(f"Validation Loss: {val_loss:.4f}")

    def on_test_end(self, metrics):
        self.logger.info("***** Test results *****")

        output_test_results_file = os.path.join(self.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info(f"{key} = {metrics[key]}")
                    writer.write(f"{key} = {metrics[key]}\n")


def main():
    # 初始化参数
    args = init_args()
    print("\n", "=" * 30, f"start", "=" * 30, "\n")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                              data_type='train', max_len=args.max_seq_length)
    logging_callback = LoggingCallback(args.output_dir)
    # 训练过程
    if args.do_train:
        print("\n******Train******")

        # 初始化 T5 模型
        model = T5FineTuner(args, T5ForConditionalGeneration.from_pretrained(args.model_name_or_path), tokenizer)
        model = model.to(device)

        train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                                    data_type='train', max_len=args.max_seq_length)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  drop_last=True, shuffle=True, num_workers=4)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = (len(train_loader.dataset) // (
                args.train_batch_size * max(1, int(args.n_gpu)))) // args.gradient_accumulation_steps * float(
            args.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

        for epoch in range(int(args.num_train_epochs)):
            train_loss = train_model(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch + 1}/{args.num_train_epochs}, Training Loss: {train_loss:.4f}")

        # 保存最终模型
        model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print("training and saving done!")

    if args.do_inference:
        print("\n******evaluate******")

        # 从之前的检查点加载 T5 模型
        print(f"Load trained model from {args.output_dir}")
        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

        model = T5FineTuner(args, tfm_model, tokenizer).to(device)

        test_dataset = ABSADataset(tokenizer, data_dir=args.dataset,
                                   data_type='test', max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
        scores = evaluate(test_loader, model, device)


if __name__ == '__main__':
    main()