import torch
from torch.nn import functional
import os

def masked_cross_entropy(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    #mask = mask.transpose(0, 1).float()
    length = torch.sum(mask, dim=-1)

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1)) ## -1 means inferred from other dimensions
    #print (logits_flat)
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat,dim=1)
    #print (log_probs_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1).long()
    # losses_flat: (batch * max_len, 1)
    #print (target_flat.size(), log_probs_flat.size())
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    #print (logits.float().sum())
    losses = losses * mask
    loss = losses.sum() / (length.float().sum() + 1e-10)
    return loss


def save_model(model, name):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/{}.bin'.format(name))


def load_model(model, name, gpu=True):
    if gpu:
        model.load_state_dict(torch.load('models/{}.bin'.format(name)))
    else:
        model.load_state_dict(torch.load('models/{}.bin'.format(name), map_location=lambda storage, loc: storage))

    return model