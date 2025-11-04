import torch.nn as nn
import torch

class BiRNNClassifier(nn.Module):
    def __init__(self,
                vocab_size, 
                hidden_dim=128, 
                rnn_type="GRU",
                num_layers=1, 
                combine="concat"):
        
        super().__init__()
        self.rnn_dict = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "RNN": nn.RNN}
        # сохраняем значение combine
        self.combine = combine
        # входная размерность эмбеддинг-слоя - vocab_size, выходная - hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, # размер словаря, число уникальных токенов
            embedding_dim=hidden_dim)
        # здесь должны быть разные блоки в зависимости от rnn_type
        self.rnn = self.rnn_dict[rnn_type](
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # out_dim может быть разным в зависимости от значения combine
        out_dim = hidden_dim * 2 if self.combine == "concat" else hidden_dim
        # выходной линейный слой
        self.fc = nn.Linear(out_dim, vocab_size)


    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        center = x.size(1) // 2

        # скрытые состояния <MSAK> токена 
        # после двух проходов двунаправленной сети
        hidden_forward = out[:, center, :out.size(2)//2]
        hidden_backward = out[:, center, out.size(2)//2:]

        # агрегация скрытых состояний в зависимости от self.combine
        hidden_agg = hidden_forward + hidden_backward\
                    if self.combine == "sum"\
                    else torch.cat((hidden_forward, hidden_backward), dim=1)

        linear_out = self.fc(hidden_agg)
        return linear_out