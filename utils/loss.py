import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationKLD(nn.Module):

    def __init__(self):
        super(KnowledgeDistillationKLD, self).__init__()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.log_softmax(teacher_logits, dim=1),
            reduction='batchmean',
            log_target=True
        )
