import torch
import torch.nn as nn
from torchvision import models

EFFICIENTNET_BUILDERS = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
}

def build_backbone(backbone_name: str):
    """
    Build a pretrained backbone that outputs feature vectors (no classifier).
    Supported:
    - densenet121      → 1024-dim features
    - efficientnet_b0  → 1280-dim features
    - efficientnet_b1  → 1280-dim features
    - efficientnet_b2  → 1408-dim features
    - efficientnet_b3  → 1536-dim features
    """
    if backbone_name == "densenet121":
        backbone = models.densenet121(weights="IMAGENET1K_V1")
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()

    elif backbone_name in EFFICIENTNET_BUILDERS:
        backbone = EFFICIENTNET_BUILDERS[backbone_name](weights="IMAGENET1K_V1")
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Supported: ['densenet121', 'efficientnet_b0', 'efficientnet_b1', "
            f"'efficientnet_b2', 'efficientnet_b3']"
        )

    return backbone, in_features

class ImageOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_name = config["backbone_name"]
        self.image_branch, in_features = build_backbone(self.backbone_name)
        for param in self.image_branch.parameters():
            param.requires_grad = False

        self.image_projector = nn.Sequential(
            nn.Linear(in_features, config["image_proj_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_proj_stage1"]),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config["image_proj_dim"], config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_fusion_stage1"]),
            nn.Linear(config["fusion_dim"], config["num_classes"]),
        )

    def set_dropout(self, dropout_proj, dropout_fusion):
        for module in self.image_projector.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_proj
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_fusion
        print(f"Dropout updated — proj: {dropout_proj}  fusion: {dropout_fusion}")
    

    def forward(self, image, meta):
        return self.classifier(self.image_projector(self.image_branch(image)))
    
class MetaOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.meta_branch = nn.Sequential(
            nn.Linear(len(config['metadata_cols']), config["meta_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_proj_stage1"]),
            nn.Linear(config["meta_hidden_dim"], config["meta_proj_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_proj_stage1"]),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config["meta_proj_dim"], config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_fusion_stage1"]),
            nn.Linear(config["fusion_dim"], config["num_classes"]),
        )

    def set_dropout(self, dropout_proj, dropout_fusion):
        for module in self.meta_branch.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_proj
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_fusion
        print(f"Dropout updated — proj: {dropout_proj}  fusion: {dropout_fusion}")
   

    def forward(self, image, meta):
        return self.classifier(self.meta_branch(meta))   

class MultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_name = config["backbone_name"]
        self.image_branch, in_features = build_backbone(self.backbone_name)
        for param in self.image_branch.parameters():
            param.requires_grad = False

        self.image_projector = nn.Sequential(
            nn.Linear(in_features, config["image_proj_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_proj_stage1"]),
        )
        self.meta_branch = nn.Sequential(
            nn.Linear(len(config['metadata_cols']), config["meta_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_proj_stage1"]),
            nn.Linear(config["meta_hidden_dim"], config["meta_proj_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_proj_stage1"]),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config["image_proj_dim"] + config["meta_proj_dim"],
                      config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_fusion_stage1"]),
            nn.Linear(config["fusion_dim"], config["num_classes"]),
        )

    def set_dropout(self, dropout_proj, dropout_fusion):
        for module in self.image_projector.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_proj
        for module in self.meta_branch.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_proj
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_fusion
        print(f"Dropout updated — proj: {dropout_proj}  fusion: {dropout_fusion}")
    
    def forward(self, image, meta):
        img_feat  = self.image_projector(self.image_branch(image))
        meta_feat = self.meta_branch(meta)
        return self.classifier(torch.cat([img_feat, meta_feat], dim=1))

def build_model(config):
    """Factory function — returns the right model based on config."""
    mode = config["fusion_mode"]
    if mode == "multimodal":
        return MultiModalModel(config)
    elif mode == "image_only":
        return ImageOnlyModel(config)
    elif mode == "meta_only":
        return MetaOnlyModel(config)
    else:
        raise ValueError(f"Unknown fusion_mode: {mode}. "
                         f"Choose from: multimodal, image_only, meta_only")