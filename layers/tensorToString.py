import torch

class FloatToStringMapper(torch.nn.Module):
    def __init__(self):
        super(FloatToStringMapper, self).__init__()

    def forward(self, x):
        # translated
        x_str = [str(value.item()) for value in x]
        my_string = ' '.join(x_str)
        return my_string
