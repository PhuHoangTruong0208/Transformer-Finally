from torch.utils.data import DataLoader, Dataset
from torch import nn 
from transformers import TranformersModel, CreateMask, device
import torch


class PreprocessingData:

    def __init__(self, limit_total_sentence, max_sequence_length, data):
        self.limit_total_sentence = limit_total_sentence
        self.max_sequence_length = max_sequence_length
        self.START_TOKEN = "<start>"
        self.END_TOKEN = "<end>"
        self.PADDING_TOKEN = "<padding>"
        self.data = data

    def read_file(self):
        with open(file=self.data, mode='r', encoding="utf-8") as file:
            data = file.read().splitlines()
        # chia dữ liệu trong file txt thành nguồn và mục tiêu
        source_sentences, target_sentences = [], []
        for i in range(len(data)):
            i = i+1
            if i % 2 == 0:
                target_sentences.append(data[i-1])
            else:
                source_sentences.append(data[i-1])
        # chuẩn hóa lại số lượng câu
        source_sentences = source_sentences[:self.limit_total_sentence]
        target_sentences = target_sentences[:self.limit_total_sentence]
        return source_sentences, target_sentences
    
    # tạo từ vựng từ các câu
    def get_vocab_index(self):
        source_sentences, target_sentences = self.read_file()
        total_sentence = source_sentences + target_sentences
        # tạo từng vựng ký tự
        vocab = [self.PADDING_TOKEN, self.END_TOKEN, self.START_TOKEN]
        for lines in total_sentence:
            for character in list(lines):
                if character not in vocab:
                    vocab.append(character)
        # tạo ra các index để mã hóa và giải mã
        index_to = {k:v for k, v in enumerate(vocab)}
        to_index = {v:k for k, v in enumerate(vocab)}
        return to_index, index_to
    
    # xử lý trước
    def preprocess(self):
        source_sentences, target_sentences = self.read_file()
        # chuẩn hóa độ dài các câu
        for i in range(len(source_sentences)):
            if len(source_sentences[i]) > self.max_sequence_length:
                source_sentences[i] = source_sentences[i][:self.max_sequence_length]

        for i in range(len(target_sentences)):
            if len(target_sentences[i]) > self.max_sequence_length:
                target_sentences[i] = target_sentences[i][:self.max_sequence_length]
        return source_sentences, target_sentences
    



class TextDataset(Dataset):

    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, idx):
        return self.source_sentences[idx], self.target_sentences[idx]
    



class TransformersTraining:

    def __init__(self, d_model=512, batch_size=30, ffn_hidden=2048, num_heads=8, dropout=0.1, num_layers=2,
                epochs=40, max_sequence_length=200, limit_total_sentence=200, data="data.txt"):
        
        self.d_model = d_model
        self.batch_size = batch_size
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers

        self.preprocess_method = PreprocessingData(limit_total_sentence, max_sequence_length, data)
        self.create_mask = CreateMask(self.preprocess_method.max_sequence_length)

        self.to_index, self.index_to = self.preprocess_method.get_vocab_index()

        self.transformer_model = TranformersModel(d_model, num_heads, ffn_hidden, dropout, self.to_index, self.preprocess_method.max_sequence_length,
                                                num_layers, self.preprocess_method.START_TOKEN, len(self.to_index), self.preprocess_method.END_TOKEN,
                                                self.preprocess_method.PADDING_TOKEN)
        
        for params in self.transformer_model.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

        self.criterian = nn.CrossEntropyLoss(ignore_index=self.to_index[self.preprocess_method.PADDING_TOKEN], reduction='none')
        self.optim = torch.optim.Adam(self.transformer_model.parameters(), lr=1e-4)

        self.epochs = epochs
        self.total_loss = 0
        self.transformer_model.train()
        self.transformer_model.to(device)
        
        self.source_sentences, self.target_sentences = self.preprocess_method.preprocess()
        dataset = TextDataset(self.source_sentences, self.target_sentences)
        self.train_loader = DataLoader(dataset, self.batch_size)

    def training(self):
        for epoch in range(self.epochs):
            print(f"Epoch : {epoch}")

            iterator = iter(self.train_loader)
            for batch_num, batch in enumerate(iterator):

                self.transformer_model.train()
                source_batch, target_batch = batch
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(source_batch, target_batch)
                self.optim.zero_grad()

                target_predict = self.transformer_model(source_batch, target_batch, encoder_self_attention_mask.to(device), decoder_self_attention_mask.to(device),
                                                        decoder_cross_attention_mask.to(device), enc_start_token=False, enc_end_token=False, dec_start_token=True,
                                                        dec_end_token=True)
                
                labels = self.transformer_model.decoder_layer.embedding.batch_tokenize(batch=target_batch, start_token=False, end_token=True)
                loss = self.criterian(
                    target_predict.view(-1, len(self.to_index)).to(device),
                    labels.view(-1).to(device)
                ).to(device)

                valid_indicies = torch.where(labels.view(-1) == self.to_index[self.preprocess_method.PADDING_TOKEN], False, True)
                loss = loss.sum() / valid_indicies.sum()
                loss.backward()
                self.optim.step()
                

                if batch_num % 100 == 0:
                    print(f"Iteration {batch_num} : {loss.item()}")
                    print("X: hello what is your name?")
                    print("Predict: ", end="", flush=True)

                    source_senttence = ("hello what is your name?",)
                    target_sentence = ("", )
                    for word_counter in range(self.preprocess_method.max_sequence_length-2):
                        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(source_senttence, target_sentence)

                        target_predict_test = self.transformer_model(source_senttence, target_sentence, encoder_self_attention_mask.to(device),
                                                        decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),
                                                        enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True)
                        
                        next_token_prob_distribution = target_predict_test[0][word_counter]
                        next_token_index = torch.argmax(next_token_prob_distribution).item()
                        next_token = self.index_to[next_token_index]
                        target_sentence = (target_sentence[0] + next_token, )
                        print(next_token, end="", flush=True)

                        if next_token == self.preprocess_method.END_TOKEN:
                            break
                    print()

    def predict(self, source_sequence):
        source_sequence = (source_sequence, )
        target_sequence = ("", )
        for word_counter in range(self.preprocess_method.max_sequence_length-2):
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(source_sequence, target_sequence)
            # dự đoán của mô hình
            target_predict = self.transformer_model(source_sequence, target_sequence, encoder_self_attention_mask.to(device),
                                                        decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),
                                                        enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True)
            
            next_token_prob_distribution = target_predict[0][word_counter]
            next_token_index = torch.argmax(next_token_prob_distribution).item()
            next_token = self.index_to[next_token_index]
            target_sequence = (target_sequence[0] + next_token, )
            print(next_token, end="", flush=True)

            if next_token == self.preprocess_method.END_TOKEN:
                break
        print()

transformers = TransformersTraining(num_heads=8, num_layers=1, d_model=64, ffn_hidden=128, dropout=0.01, epochs=100,
                            limit_total_sentence=727)
transformers.training()
while True:
    user_input = input("bạn : ")
    predict = transformers.predict(source_sequence=user_input)