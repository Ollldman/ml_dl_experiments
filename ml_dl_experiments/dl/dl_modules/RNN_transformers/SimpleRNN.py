import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SimpleRNN(nn.Module):
    def __init__(self,
                vocab_size,
                embedding_dim, 
                hidden_size, 
                output_size, 
                num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )


    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        lengths = lengths.cpu()
        packed = pack_padded_sequence(
            input=embedded,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        
        # Используем последнее скрытое состояние для классификации
        out = self.fc1(hidden[-1])
        return out