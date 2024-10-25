from typing import Dict, List, Union

import torch
import timm


class SiameseNetwork(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        
        # Backbone from timm package
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.backbone, 'fc'):
            # e.g., ResNet-like models
            self.backbone.fc = torch.torch.nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            # e.g., EfficientNet-like models
            self.backbone.classifier = torch.torch.nn.Identity()

        # Cosine similarity layer
        self.similarity = torch.torch.nn.CosineSimilarity(dim=1)
        
        # To store precomputed embeddings
        self.class_embeddings: Union[Dict, None] = None
        
    def forward_once(self, x):
        """Forward pass Siamese network once."""
        return self.backbone(x)   

    def precompute_class_embeddings(self, reference_images_per_class: Dict[str, List]):
        """
        Precompute embeddings for the reference images of each class.
        
        Args:
            reference_images_per_class (dict): Dictionary where each key is a class label
                                               and the value is a list of N reference images
                                               (as tensors) for that class.
        """        
        self.class_embeddings = {} # Reset class embeddings
        for class_label, images in reference_images_per_class.items():
            with torch.no_grad():
                embeddings = [self.forward_once(image.unsqueeze(0)) for image in images]  # Compute embedding for each reference image
                embeddings = torch.stack(embeddings)  # Stack embeddings to create a tensor
                self.class_embeddings[class_label] = embeddings # Store embeddings per class
    
    def forward(self, img1, img2=None):
        """
        Forward pass for the Siamese network.
        
        Args:
            img1: A single test image or batch of test images (tensor).
            img2: Optional. Used for direct comparison between two images.                  
                  If None, will compare img1 to precomputed class embeddings.
        
        Returns:
            Similarity score if img2 is None, otherwise a dictionary of similarity
            scores against each class.
        """
        if img2 is not None:            
            embedding1 = self.forward_once(img1)
            embedding2 = self.forward_once(img2)
            similarity_score = self.similarity(embedding1, embedding2)
            return similarity_score
        
        else:
            if self.class_embeddings is None:
                raise ValueError("Class embeddings have not been precomputed. Run precompute_class_embeddings first.")
            
            embedding1 = self.forward_once(img1)  # Get embedding for test image(s)
            similarity_scores = {}
            
            for class_label, class_embedding in self.class_embeddings.items():
                num_ref = class_embedding.shape[0]
                class_embedding = class_embedding.permute(1, 2, 0)                
                similarity_scores[class_label] = self.similarity(
                    embedding1.unsqueeze(2).repeat(1, 1, num_ref), class_embedding
                ).cpu().numpy()
            
            return similarity_scores
