import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model onfiguration with static class attributes."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ration = 1 / 3
    batch_size = 8
    fh = 1
    input_size = 5
    r = 2
    graph_cells = 9
    scaler_type = "standard"
    data_path = "../../data2019-2021_BIG.grib"
    random_state = 42
    input_dims = (32, 48)
    output_dims = (25, 45)
