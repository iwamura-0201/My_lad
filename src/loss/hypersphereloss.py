import torch.nn as nn
import torch


class HyperSphereLoss(nn.Module):
    """
    HyperSphereLoss with numerical stability improvements.
    Computes variance of cls_output with proper handling of edge cases.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # Small epsilon for numerical stability

    def forward(self, output, data):
        cls_output = output["cls_output"]  # [batch_size, hidden_dim]
        
        # Check for NaN/Inf in input - これはモデル出力の問題
        if torch.isnan(cls_output).any():
            print("⚠️ Warning: NaN detected in cls_output (model output issue)")
            print(f"   NaN count: {torch.isnan(cls_output).sum().item()}")
            # NaNが入力されている = モデルに問題がある
            # ゼロロスを返して学習を継続させない
            return torch.tensor(float('nan'), device=cls_output.device, requires_grad=True)
        
        if torch.isinf(cls_output).any():
            print("⚠️ Warning: Inf detected in cls_output (model output issue)")
            print(f"   Inf count: {torch.isinf(cls_output).sum().item()}") 
            return torch.tensor(float('nan'), device=cls_output.device, requires_grad=True)
        
        # Check batch size
        batch_size = cls_output.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=cls_output.device, requires_grad=True)
        
        # HyperSphereLossの本来の目的:
        # cls_outputの各次元の分散を計算し、それを最小化
        # これによりhypersphere上に埋め込まれる
        
        # 各特徴次元の分散を計算 (dim=0でバッチ方向に平均化)
        variance = torch.var(cls_output, dim=0, unbiased=True)  # [hidden_dim]
        
        # 全次元の分散の平均
        mean_variance = variance.mean()
        
        # Final check
        if torch.isnan(mean_variance):
            print("⚠️ Warning: NaN variance in HyperSphereLoss")
            return torch.tensor(float('nan'), device=cls_output.device, requires_grad=True)
        
        return mean_variance
