import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int = 2, d_model: int = 32, nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out = self.transformer(x)
        out = self.fc_out(out[:, -1, :])
        return out

class TransformerForecaster:
    def __init__(self):
        self.model = TransformerModel()
        self.model.eval()

    def predict_next(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        sequence: Array of shape (seq_len, 2) containing [temperature, vibration]
        """
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            pred = self.model(x).squeeze(0).numpy()
            
            # Apply physics-constrained smooth forecast step
            last_temp, last_vib = sequence[-1][0], sequence[-1][1]
            pred_temp = float(last_temp + np.clip(pred[0] * 0.1, -1.0, 2.0))
            pred_vib = float(max(0.05, last_vib + np.clip(pred[1] * 0.01, -0.05, 0.05)))
            
            return pred_temp, pred_vib