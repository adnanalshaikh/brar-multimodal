from torchvision import transforms

def get_transforms(config, train=True):
    version = config.get('transform_version', 'default')

    base = [
        #PadToSquare(),
        transforms.Resize((224, 224)), #224×112
    ]

    if train:
        if version == 'v1':
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomAutocontrast(p=0.2),
            ]
        elif version == 'v2':
            aug = []
        else:
            aug = [
                transforms.RandomRotation(2),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            ]

        return transforms.Compose(
            base + aug + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        base + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

