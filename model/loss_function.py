import torch
from torch import nn
import torchaudio.functional as F


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    @staticmethod
    def forward(model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(
            mel_out_postnet, mel_target
        )
        # mel_loss = nn.MSELoss()(mel_out, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # delta loss
        # delta = F.compute_deltas(mel_out_postnet)
        # delta2 = F.compute_deltas(delta)
        # mel_target_delta = F.compute_deltas(mel_target)
        # mel_target_delta2 = F.compute_deltas(mel_target_delta)
        # delta_loss = nn.MSELoss()(delta, mel_target_delta) + nn.MSELoss()(delta2, mel_target_delta2)

        loss = mel_loss + gate_loss
        # loss = mel_loss + gate_loss + delta_loss

        return loss, mel_loss, gate_loss


class EdmLoss(nn.Module):
    def __init__(self):
        super(EdmLoss, self).__init__()

    @staticmethod
    def forward(edm_out, targets):
        """
        return:
            L_sc -- speaker classification loss
            L_ec -- style classification loss
        """
        emo_emb, emo_cls, spk_emb, spk_cls, r_emo_cls = edm_out
        _, emo_lab, spk_lab = targets  # [N]

        emo_loss = nn.CrossEntropyLoss()(emo_cls, emo_lab)
        spk_loss = nn.CrossEntropyLoss()(spk_cls, spk_lab)
        r_emo_loss = nn.CrossEntropyLoss()(r_emo_cls, emo_lab)
        ort_loss = ort_loss(emo_emb, spk_emb)

        edm_loss = emo_loss + spk_loss + r_emo_loss + ort_loss

        return edm_loss, emo_loss, spk_loss, r_emo_loss, ort_loss

    def ort_loss(self, features_1, features_2):
        dot_product = torch.matmul(features_1, features_2.t())
        mean_dot_product = torch.mean(dot_product)

        loss = mean_dot_product**2

        return loss
