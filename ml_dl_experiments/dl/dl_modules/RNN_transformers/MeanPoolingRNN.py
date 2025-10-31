import torch.nn as nn

class MeanPoolingRNN(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            emb_dim=300, 
            hidden_dim=256, 
            num_layers=1, 
            output_dim=2, 
            pad_idx=0, 
            dropout_p=0.3):
        
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 
            emb_dim, 
            padding_idx=pad_idx)
        
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(p=dropout_p)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        #  xavier инициализацию весов
        print("Start weighs and biases init")
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
                print(f"Success Xavier uniform init to {name}")
            elif "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
                print(f"Success Xavier uniform init to {name}")
            elif "weight_ih" in name:
                nn.init.zeros_(param.data)
                print(f"Success zeros init to {name}")

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        rnn_out, _ = self.rnn(x)

        rnn_out_normed = self.norm(rnn_out)

        # mean pooling по attention_mask
        mask = attention_mask.unsqueeze(-1).expand_as(rnn_out_normed)
        masked_out = mask * rnn_out_normed
        summed = masked_out.sum(dim=1)
        lengths = attention_mask.sum(dim=1).unsqueeze(1).clamp(min=1)
        mean_pooled = summed / lengths

        out = self.dropout(mean_pooled)
        logits = self.fc(out)

        return logits