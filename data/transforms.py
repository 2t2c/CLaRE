import albumentations as A


def get_clip_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.5,
            ),
            A.RandomRotate90(p=0.33),
        ],
        p=1.0,
    )
