import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.utils.class_weight import compute_class_weight



def freeze_backbone(model):
    """Freeze all image-backbone parameters."""
    for param in model.image_branch.parameters():
        param.requires_grad = False

def unfreeze_stage2_backbone(model):
    """
    Unfreeze last blocks of backbone for stage 2 fine-tuning.
    """
    backbone_name = model.backbone_name

    if backbone_name == "densenet121":
        # Unfreeze denseblock4 AND denseblock3
        blocks_to_unfreeze = ['denseblock4', 'denseblock3', 'norm5', 'transition3']
        for block_name in blocks_to_unfreeze:
            block = getattr(model.image_branch.features, block_name, None)
            if block:
                for param in block.parameters():
                    param.requires_grad = True
                
    elif backbone_name.startswith("efficientnet"):
        # Unfreeze last 4 blocks for all EfficientNet variants
        blocks = model.image_branch.features
        num_blocks_to_unfreeze = min(4, len(blocks))
        
        for param in blocks[-num_blocks_to_unfreeze:].parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unsupported backbone for stage-2 unfreezing: {backbone_name}")

def unfreeze_all(model):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True

def _get_train_labels(train_loader):
    """Safely get labels from dataset, handling Subset wrapping."""
    dataset = train_loader.dataset
    if hasattr(dataset, 'df'):
        return dataset.df["level"].values
    elif hasattr(dataset, 'dataset'):  # Subset wrapper
        return dataset.dataset.df["level"].values
    else:
        raise AttributeError("Cannot find df attribute in dataset or its parent")

def get_criterion(config, train_loader, device):
    if config['weight_mode'] == 'balanced':
        train_labels = _get_train_labels(train_loader) - 1
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1, 2]),
            y=train_labels
        )
        weights = torch.tensor(weights, dtype=torch.float32)

    elif config['weight_mode'] == 'manual':
        weights = torch.tensor(config['class_weights'], dtype=torch.float32)

    elif config['weight_mode'] in ('fourth_root', 'square_root'):
        train_labels = _get_train_labels(train_loader)
        class_counts = torch.tensor([
            (train_labels == 1).sum(),
            (train_labels == 2).sum(),
            (train_labels == 3).sum(),
        ], dtype=torch.float32)
        exp = 0.25 if config['weight_mode'] == 'fourth_root' else 0.5
        weights = 1.0 / (class_counts ** exp)

    else:
        weights = None

    if weights is not None:
        weights = weights / weights.sum()
        print(f"Class weights ({config['weight_mode']}) — "
              f"L1: {weights[0]:.4f}  L2: {weights[1]:.4f}  L3: {weights[2]:.4f}")

    return nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=config['label_smoothing']
    )

def _build_scheduler(optimizer, monitor, config):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if monitor == "val_loss" else "max",
        patience=config.get('scheduler_patience', 5),
        factor=config.get('scheduler_factor', 0.5),
        min_lr=1e-8,
    )

def _append_history(full, hist):
    full["train_loss"] += hist["train_loss"]
    full["val_loss"]   += hist["val_loss"]
    full["train_acc"]  += hist["train_acc"]
    full["val_acc"]    += hist["val_acc"]
    full["stage_boundaries"].append(len(full["train_loss"]))
    return full

def _run_epochs(model, train_loader, val_loader, optimizer, criterion,
                num_epochs, device, stage, scheduler,
                best_state, best_metric, epochs_no_improve,
                patience, monitor, train_verbose = False):

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

# Note: for meta_only mode, images is a zero tensor — moved to device
# for API consistency but not used in forward pass

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for images, meta, labels in train_loader:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct  += (outputs.argmax(dim=1) == labels).sum().item()
            train_total    += batch_size

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, meta, labels in val_loader:
                images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                outputs = model(images, meta)
                loss = criterion(outputs, labels)
                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct  += (outputs.argmax(dim=1) == labels).sum().item()
                val_total    += batch_size

        avg_train_loss = train_loss_sum / train_total
        avg_val_loss   = val_loss_sum   / val_total
        train_acc      = train_correct  / train_total
        val_acc        = val_correct    / val_total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_metric = avg_val_loss if monitor == "val_loss" else val_acc
        scheduler.step(current_metric)

        improved = (current_metric < best_metric if monitor == "val_loss"
                    else current_metric > best_metric)
        if improved:
            best_metric = current_metric
            best_state  = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            mark = "*"
        else:
            epochs_no_improve += 1
            mark = ""

        if train_verbose:
            print(f"Stage {stage} | Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} {mark}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at Stage {stage}, Epoch {epoch+1}")
            return best_state, best_metric, epochs_no_improve, True, history

    return best_state, best_metric, epochs_no_improve, False, history

def train(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = get_criterion(config, train_loader, device)
    monitor = config["monitor"]

    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("inf") if monitor == "val_loss" else -float("inf")

    full_history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "stage_boundaries": [],
    }

    train_verbose = config.get('train_verbose', False)
    
    # -------------------------------  meta only
    # meta_only — single stage, no backbone to unfreeze
    if config['fusion_mode'] == 'meta_only':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr_stage1'],
            weight_decay=config['wd_stage1']
        )
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode="min" if monitor == "val_loss" else "max",
        #    patience=5, factor=0.5, min_lr=1e-6
        #)
        
        scheduler = _build_scheduler(optimizer, monitor, config)
        
        
        best_state, best_metric, _, _, hist = _run_epochs(
            model, train_loader, val_loader, optimizer, criterion,
            config['num_epochs_1'], device, stage=1, scheduler=scheduler,
            best_state=best_state, best_metric=best_metric,
            epochs_no_improve=0, patience=config['patience_1'], monitor=monitor, train_verbose = train_verbose
        )
        full_history = _append_history(full_history, hist)
        model.load_state_dict(best_state)
        return model, full_history
    
    # ------------------------------- Stage 1: frozen backbone
    freeze_backbone(model)
    model.set_dropout(config['dropout_proj_stage1'], config['dropout_fusion_stage1'])
    
    params = list(model.image_projector.parameters()) + list(model.classifier.parameters())
    
    if hasattr(model, 'meta_branch'):
        params += list(model.meta_branch.parameters())
    
    optimizer = torch.optim.Adam(params, lr=config["lr_stage1"],
                                 weight_decay=config["wd_stage1"])

    scheduler = _build_scheduler(optimizer, monitor, config)

    best_state, best_metric, _, _, hist = _run_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config["num_epochs_1"],
        device=device,
        stage=1,
        scheduler=scheduler,
        best_state=best_state,
        best_metric=best_metric,
        epochs_no_improve=0,
        patience=config["patience_1"],
        monitor=monitor,
        train_verbose = train_verbose
    )
    full_history = _append_history(full_history, hist)
    model.load_state_dict(best_state)

    # ------------------------------- Stage 2: unfreeze final backbone block
    unfreeze_stage2_backbone(model)
    model.set_dropout(config['dropout_proj_stage2'], 
                      config['dropout_fusion_stage2'])
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr= config["lr_stage2"], weight_decay=config["wd_stage2"]
    )
    scheduler = _build_scheduler(optimizer, monitor, config)
    
    best_state, best_metric, _, _, hist = _run_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config["num_epochs_2"],
        device=device,
        stage=2,
        scheduler=scheduler,
        best_state=best_state,
        best_metric=best_metric,
        epochs_no_improve=0,
        patience=config["patience_2"],
        monitor=monitor,
        train_verbose = train_verbose
    )
    full_history = _append_history(full_history, hist)
    model.load_state_dict(best_state)

    # ------------------------------- Stage 3: unfreeze entire model
    unfreeze_all(model)
    model.set_dropout(config['dropout_proj_stage3'], config['dropout_fusion_stage3'])
        
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr_stage3"],
        weight_decay=config["wd_stage3"],
    )
    scheduler = _build_scheduler(optimizer, monitor, config)

    best_state, best_metric, _, _, hist = _run_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config["num_epochs_3"],
        device=device,
        stage=3,
        scheduler=scheduler,
        best_state=best_state,
        best_metric=best_metric,
        epochs_no_improve=0,
        patience=config["patience_3"],
        monitor=monitor,
        train_verbose = train_verbose
    )
    full_history = _append_history(full_history, hist)
    model.load_state_dict(best_state)

    return model, full_history

