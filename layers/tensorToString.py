import torch

class FloatToStringMapper(torch.nn.Module):
    """Utility module that converts a 1-D float tensor into a single space-separated string."""
    def __init__(self):
        super(FloatToStringMapper, self).__init__()

    def forward(self, x):
        x_str = [str(value.item()) for value in x]
        my_string = ' '.join(x_str)
        return my_string
