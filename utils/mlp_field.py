import torch
import torch.nn as nn


# DISCLAIMER: the following is mostly ChatGPT
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        """
        Args:
            num_freqs: number of frequency bands (K).
            include_input: if True, prepend the raw x to the encoding.
            log_sampling: if True, use 2^k bands, else linearly spaced.
        """
        super().__init__()
        self.include_input = include_input
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D_in]
        returns: [B, D_in * (include_input + 2 * num_freqs)]
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


class MlpField(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        pe_freqs: int = 6,
        pe_include_input: bool = True,
    ):
        """
        Args:
            input_dim: Number of input dimensions (D_in).
            output_dim: Number of output dimensions (D_out).
        """
        super().__init__()
        self.pe = PositionalEncoding(
            num_freqs=pe_freqs,
            include_input=pe_include_input,
            log_sampling=True,
        )

        encoded_dim = input_dim * (int(pe_include_input) + 2 * pe_freqs)

        layers = [nn.Linear(encoded_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D_in] input tensor
        Returns: [B, D_out]
        """
        x_enc = self.pe(x)
        return self.net(x_enc)


def normalized_pixel_grid(height, width, device="cpu"):
    y = torch.linspace(0, 1, steps=height, device=device)
    x = torch.linspace(0, 1, steps=width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # shape: [H, W]
    grid = torch.stack((xx, yy), dim=-1)  # shape: [H, W, 2]
    return grid  # Each entry is (x, y) in [0, 1]
