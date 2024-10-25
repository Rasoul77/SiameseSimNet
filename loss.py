import torch


class ContrastiveLossCosine(torch.nn.Module):
    """Loss function based on Cosine Similarity."""
    def __init__(self, margin=0.5):
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin
    
    def forward(self, similarity_score, label):   
        # (score, label) = (1, 1) -> pos_loss = 0.0 and neg_loss = 0.0 -> loss = 0.0     
        # (score, label) = (1, 0) -> pos_loss = 0.0 and neg_loss = 0.5 -> loss = 0.5
        # (score, label) = (0, 1) -> pos_loss = 1.0 and neg_loss = 0.0 -> loss = 1.0
        # (score, label) = (0, 0) -> pos_loss = 0.0 and neg_loss =-0.5 -> loss = 0.0
        # The loss function punishes more the incorrect classification of similar pairs
        pos_loss = (1 - similarity_score) * label 
        neg_loss = (similarity_score - self.margin) * (1 - label)
        loss = pos_loss + torch.relu(neg_loss)
        return loss.mean()
