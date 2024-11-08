import torch
import torch.nn as nn
import torch.nn.functional as F
  
class LSAALoss(nn.Module):
    def __init__(self, num_iters=20, tau=1.0):
        super().__init__()
        self.num_iters = num_iters
        self.tau = tau  # Temperature parameter to control the sharpness of the approximation

    def forward(self, predictions, targets):
        batch_size, sequence_length, vocab_size = predictions.size()
        
        # Flatten the predictions and targets
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Compute the cross-entropy loss for each prediction-target pair
        ce_loss = F.cross_entropy(predictions_flat, targets_flat, reduction='none')
        ce_loss = ce_loss.view(batch_size, sequence_length, -1)

        # Compute the matchings using the Sinkhorn-Knopp algorithm
        # Initial cost matrix is the negative cross-entropy loss
        cost_matrix = -ce_loss / self.tau

        # Make cost matrix square by padding with large negative values (which become zero probabilities)
        if sequence_length < vocab_size:
            cost_matrix = F.pad(cost_matrix, (0, vocab_size - sequence_length), "constant", -1e9)
        elif sequence_length > vocab_size:
            cost_matrix = cost_matrix[:, :, :vocab_size]

        # Sinkhorn iterations to make the matrix doubly stochastic
        for _ in range(self.num_iters):
            # Row normalization
            cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=2, keepdim=True)
            # Column normalization
            cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=1, keepdim=True)

        # Exponentiate the log-cost matrix to get probabilities
        matching_matrix = torch.exp(cost_matrix)

        # Compute the loss as the sum of the element-wise product of the cost and the matching matrix
        loss = torch.sum(matching_matrix * ce_loss, dim=(1, 2)).mean()

        return loss

