"""Only supervised deep segmentation and EMA-teacher probability consistency."""
import torch
import torch.nn.functional as F


def supervised_loss(outputs, labels):
    ce_terms, dice_terms = [], []
    for output in outputs:
        target = F.interpolate(labels[:, None].float(), size=output.shape[2:],
                               mode="nearest")[:, 0].long()
        ce_terms.append(F.cross_entropy(output, target))
        foreground = output.softmax(dim=1)[:, 1]
        intersection = 2 * (foreground * target).sum() + 1e-8
        denominator = foreground.sum() + target.sum() + 1e-8
        dice_terms.append(1 - intersection / denominator)
    ce, dice = torch.stack(ce_terms).mean(), torch.stack(dice_terms).mean()
    return (ce + dice) / 2, ce, dice


def consistency_loss(student_outputs, teacher_outputs):
    if len(student_outputs) != len(teacher_outputs):
        raise ValueError("Student/teacher scales must match")
    return torch.stack([
        F.mse_loss(student.softmax(dim=1), teacher.detach().softmax(dim=1))
        for student, teacher in zip(student_outputs, teacher_outputs)
    ]).mean()


@torch.no_grad()
def update_ema(student, teacher, decay, global_step):
    decay = min(1 - 1 / (global_step + 1), decay)
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1 - decay)
    # Preserve DuCiSC's BN policy: teacher train-mode buffers are updated by its
    # own unlabeled forward, while only trainable model parameters use EMA.
