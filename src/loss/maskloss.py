import torch.nn as nn
import torch.nn.functional as F


class MaskLoss(nn.Module):
    """
    マスク言語モデルのLoss計算。
    
    大きなvocab_sizeでPyTorchのNLLLoss2dカーネルがCUDAエラーを引き起こす問題を
    回避するため、テンソルをフラット化してからF.nll_lossを使用する。
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, data):
        # output["logkey_output"]: [B, S, V] (log確率)
        # data["bert_label"]: [B, S] (ラベル)
        logkey_output = output["logkey_output"]
        bert_label = data["bert_label"]
        
        batch_size, seq_len, vocab_size = logkey_output.shape
        
        # フラット化: [B, S, V] -> [B*S, V], [B, S] -> [B*S]
        logkey_output_flat = logkey_output.contiguous().view(-1, vocab_size)
        bert_label_flat = bert_label.contiguous().view(-1)
        
        # F.nll_lossを使用（ignore_index=0でpadding位置を無視）
        return F.nll_loss(logkey_output_flat, bert_label_flat, ignore_index=0, reduction='mean')

