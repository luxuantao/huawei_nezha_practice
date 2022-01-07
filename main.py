import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from NEZHA.model_nezha import BertConfig, BertForSequenceClassification
from NEZHA import nezha_utils
import numpy as np
from transformers import BertTokenizer, AdamW, AutoModel, AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':
    pre_model_dir = 'NEZHA/nezha-base-wwm/'  # 填写预训练模型所在目录
    tokenizer = BertTokenizer(os.path.join(pre_model_dir, 'vocab.txt'), do_lower_case=True)

    data = []
    with open('data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            label, text = int(line[0]), line[2:]
            text = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True)
            data.append((text, label))
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.8)]
    test_data = data[len(train_data):]

    input_ids_train = torch.LongTensor([each[0]['input_ids'] for each in train_data]).to(device)
    token_type_ids_train = torch.LongTensor([each[0]['token_type_ids'] for each in train_data]).to(device)
    attention_mask_train = torch.LongTensor([each[0]['attention_mask'] for each in train_data]).to(device)
    label_train = torch.LongTensor([each[1] for each in train_data]).to(device)

    input_ids_test = torch.LongTensor([each[0]['input_ids'] for each in test_data]).to(device)
    token_type_ids_test = torch.LongTensor([each[0]['token_type_ids'] for each in test_data]).to(device)
    attention_mask_test = torch.LongTensor([each[0]['attention_mask'] for each in test_data]).to(device)
    label_test = torch.LongTensor([each[1] for each in test_data]).to(device)

    train_dataset = TensorDataset(input_ids_train, token_type_ids_train, attention_mask_train, label_train)
    test_dataset = TensorDataset(input_ids_test, token_type_ids_test, attention_mask_test, label_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    Bert_config = BertConfig.from_json_file(os.path.join(pre_model_dir, 'config.json'))
    model = BertForSequenceClassification(config=Bert_config, num_labels=2)
    nezha_utils.torch_init_model(model, os.path.join(pre_model_dir, 'pytorch_model.bin'))

    # model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        print('epoch:', epoch)
        model.train()
        for input_ids, token_type_ids, attention_mask, labels in tqdm(train_loader):
            optimizer.zero_grad()
            # loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels).loss
            loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0
        acc = 0
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, labels in tqdm(test_loader):
                # logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
                logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                pred = torch.argmax(logits, dim=-1)
                total += pred.size(0)
                acc += pred.eq(labels).sum().item()
        print(acc / total)
