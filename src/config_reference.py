from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
RUNS_DIR     = PROJECT_ROOT / 'runs'

base_config = {
    'run_id': 'multi_dens',  # prefix for run directory and files
    'seed' : [42, 123, 456, 789, 2024, 0, 1, 2, 3, 4, 99, 777, 314, 100, 200, 500, 1000, 2000, 3000, 9999], #7, 13, 50, 250, 1234, 5678, 12345, 54321, 99999, 31415],

    # Data
    'val_size':  0.15,
    'test_size': 0.15,

    # Augmentation
    'transform_version': 'v1',  # 'default', 'v1', 'v2'

    # Input
    'input_size': (224, 224),  # (height, width)

    # Training
    'num_epochs':   50,        # fallback if stage-specific not set
    'batch_size':   32,
    'monitor':      'val_loss',  # 'val_loss' or 'val_acc'
    'num_workers':  2,

    # Stage-specific epochs
    'num_epochs_1': 50,  # 50
    'num_epochs_2': 60,  # 60
    'num_epochs_3': 40,  # 40

    # Stage-specific patience
    'patience_1': 8,
    'patience_2': 10,
    'patience_3': 10,

    # Stage 1
    'lr_stage1': 1e-3,
    'wd_stage1': 1e-5,
    'dropout_proj_stage1':   0.15, #0.15,
    'dropout_fusion_stage1': 0.25, #0.25,

    # Stage 2
    'lr_stage2': 2e-5,
    'wd_stage2': 2e-4,
    'dropout_proj_stage2':   0.45,
    'dropout_fusion_stage2': 0.60,

    # Stage 3
    'lr_stage3': 5e-6,
    'wd_stage3': 1e-5,
    'dropout_proj_stage3':   0.45,
    'dropout_fusion_stage3': 0.60,

    # Scheduler
    'scheduler_patience': 5,
    'scheduler_factor':   0.5,

    # Loss
    'label_smoothing': 0.0,
    'weight_mode':     'fourth_root',  # 'none', 'balanced', 'manual', 'fourth_root', 'square_root'
    'class_weights':   [2.0, 1.0, 2.0],  # used only when weight_mode='manual'

    # Model
    'backbone_name':   'densenet121',  # 'densenet121', 'efficientnet_b0/b1/b2/b3'
    'fusion_mode':     'multimodal',   # 'multimodal', 'image_only', 'meta_only'
    'image_proj_dim':  64,
    'meta_hidden_dim': 32,
    'meta_proj_dim':   64,
    'fusion_dim':      64,
    'num_classes':     3,
    
    'metadata_cols': [
    "age",
    "gender", 
    "number_of_missing_teeth",
    "implant",
    "residual_root",
    "functional_tooth_logarithm"
    ], # 
    
    'normalize_cols': [
        "age",
        "number_of_missing_teeth",
        "implant",
        "residual_root",
        "functional_tooth_logarithm"
    ],

    'train_verbose' : False,
}

config = copy.deepcopy(base_config)


