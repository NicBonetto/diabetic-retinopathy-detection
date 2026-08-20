import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from dr_detection.core.data import DEFAULT_TARGET_RADIUS, preprocess_fundus


def create_train_val_split(
    labels_df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits."""
    label_col = 'diagnosis'

    if test_size > 0: 
        train_val_df, test_df = train_test_split(
            labels_df,
            test_size=test_size,
            stratify=labels_df[label_col],
            random_state=random_state
        )

        adjusted_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df[label_col],
            random_state=random_state
        )
    else:
        train_df, val_df = train_test_split(
            labels_df,
            test_size=val_size,
            stratify=labels_df[label_col],
            random_state=random_state
        )
        test_df = pd.DataFrame()

    return train_df, val_df, test_df

def create_cv_folds(
    labels_df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build stratified K-fold splits.

    A single 20% validation split of ~3.7k images leaves ~730 validation
    samples, where kappa differences of a few hundredths are indistinguishable
    from split noise. K-fold lets you average across folds instead.
    """
    label_col = 'diagnosis'
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in skf.split(labels_df, labels_df[label_col]):
        folds.append((labels_df.iloc[train_idx], labels_df.iloc[val_idx]))

    return folds


def copy_images(
    df: pd.DataFrame,
    source_dir: Path,
    dest_dir: Path,
    image_col: str,
    preprocess: bool = False,
    target_radius: int = DEFAULT_TARGET_RADIUS,
    ben_graham: bool = True
) -> None:
    """
    Copy images to the destination directory, optionally baking in the
    fundus preprocessing pipeline.

    Baking it in here means the expensive work (radius rescale plus a
    full-resolution Gaussian blur) happens once rather than on every
    __getitem__ of every epoch.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    verb = 'Preprocessing' if preprocess else 'Copying'
    print(f'{verb} {len(df)} images to {dest_dir}')

    copied = 0
    missing = []

    for image_id in tqdm(df[image_col], desc=f'{verb} images'):
        source_file = source_dir / f'{image_id}.png'

        if not source_file.is_file():
            missing.append(image_id)
            continue

        dest_file = dest_dir / source_file.name

        if preprocess:
            image = np.array(Image.open(source_file).convert('RGB'))
            image = preprocess_fundus(
                image,
                target_radius=target_radius,
                apply_ben_graham=ben_graham
            )
            Image.fromarray(image).save(dest_file)
        else:
            shutil.copy2(source_file, dest_file)

        copied += 1

    print(f'Copied: {copied}')

    if missing:
        preview = ', '.join(str(m) for m in missing[:5])
        suffix = ', ...' if len(missing) > 5 else ''
        print(f'WARNING: {len(missing)} image(s) listed in the labels were '
              f'not found and were skipped: {preview}{suffix}')

def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str
):
    """Print statistics about the data splits."""
    class_names = {
        0: 'No DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferative'
    }
    
    print("\n" + "="*60)
    print("DATA SPLIT STATISTICS")
    print("="*60)
    
    print(f"\nTotal samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Training:   {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    if len(test_df) > 0:
        print(f"  Test:       {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    
    # Class distribution
    print("\nClass Distribution:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10}", end='')
    if len(test_df) > 0:
        print(f"{'Test':<10}")
    else:
        print()
    print("-" * 60)
    
    for class_id in sorted(train_df[label_col].unique()):
        class_name = class_names.get(class_id, f'Class {class_id}')
        train_count = (train_df[label_col] == class_id).sum()
        val_count = (val_df[label_col] == class_id).sum()
        
        print(f"{class_name:<20} {train_count:<10} {val_count:<10}", end='')
        
        if len(test_df) > 0:
            test_count = (test_df[label_col] == class_id).sum()
            print(f"{test_count:<10}")
        else:
            print()
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess diabetic retinopathy dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing raw images'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to labels CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Validation set size (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.0,
        help='Test set size (default: 0.0 = no test set)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='Copy images to output directory (default: just create label files)'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=0,
        help='Number of stratified CV folds. 0 (default) keeps the single '
             'train/val split; 5 is recommended for this dataset size'
    )
    parser.add_argument(
        '--preprocess-images',
        action='store_true',
        help='Bake the fundus pipeline (radius rescale, Ben Graham, circular '
             'crop) into the copied images. Implies --copy-images. Set '
             'data.preprocess.enabled to false in the training config after.'
    )
    parser.add_argument(
        '--target-radius',
        type=int,
        default=DEFAULT_TARGET_RADIUS,
        help='Retina radius in pixels to normalise every image to'
    )
    parser.add_argument(
        '--no-ben-graham',
        action='store_true',
        help='Skip the Ben Graham local-average subtraction when preprocessing'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("DIABETIC RETINOPATHY DATA PREPROCESSING")
    print("="*60)
    print(f"Source directory: {data_dir}")
    print(f"Labels file: {args.labels}")
    print(f"Output directory: {output_dir}")
    print(f"Validation size: {args.val_size*100:.1f}%")
    if args.test_size > 0:
        print(f"Test size: {args.test_size*100:.1f}%")
    print(f"Random seed: {args.seed}")
    print()

    print("Loading labels...")
    labels_df = pd.read_csv(args.labels)
    print(f"Loaded {len(labels_df)} samples")

    image_col = 'id_code'
    label_col = 'diagnosis'
    print(f"Using columns: image_id='{image_col}', label='{label_col}'")
    
    test_df = pd.DataFrame()
    pool_df = labels_df

    if args.test_size > 0:
        pool_df, test_df = train_test_split(
            labels_df,
            test_size=args.test_size,
            stratify=labels_df[label_col],
            random_state=args.seed
        )
        print(f"\nHeld out {len(test_df)} samples as a test set")

    output_dir.mkdir(parents=True, exist_ok=True)

    def standardise(df):
        return df.rename(columns={image_col: 'image_id', label_col: 'diagnosis'})

    if args.folds >= 2:
        print(f"\nCreating {args.folds} stratified CV folds...")
        folds = create_cv_folds(pool_df, n_folds=args.folds, random_state=args.seed)

        for i, (train_df, val_df) in enumerate(folds):
            print(f"\n--- Fold {i} ---")
            print_split_statistics(train_df, val_df, pd.DataFrame(), label_col)

            standardise(train_df).to_csv(
                output_dir / f'train_labels_fold{i}.csv', index=False)
            standardise(val_df).to_csv(
                output_dir / f'val_labels_fold{i}.csv', index=False)
            print(f"  Saved: train_labels_fold{i}.csv / val_labels_fold{i}.csv")
    else:
        print("\nCreating train/val split...")
        train_df, val_df = train_test_split(
            pool_df,
            test_size=args.val_size,
            stratify=pool_df[label_col],
            random_state=args.seed
        )

        print_split_statistics(train_df, val_df, test_df, label_col)

        standardise(train_df).to_csv(output_dir / 'train_labels.csv', index=False)
        standardise(val_df).to_csv(output_dir / 'val_labels.csv', index=False)
        print(f"  Saved: {output_dir / 'train_labels.csv'}")
        print(f"  Saved: {output_dir / 'val_labels.csv'}")

    if len(test_df) > 0:
        standardise(test_df).to_csv(output_dir / 'test_labels.csv', index=False)
        print(f"  Saved: {output_dir / 'test_labels.csv'}")

    bake = args.preprocess_images
    want_images = args.copy_images or bake

    if want_images and args.folds >= 2:
        image_dir = output_dir / 'images'
        print(f"\nWriting a single image directory shared by all folds: {image_dir}")
        copy_images(
            labels_df, data_dir, image_dir, image_col,
            preprocess=bake,
            target_radius=args.target_radius,
            ben_graham=not args.no_ben_graham
        )
    elif want_images:
        print("\nWriting images to output directory...")
        for split_df, name in ((train_df, 'train'), (val_df, 'val')):
            copy_images(
                split_df, data_dir, output_dir / name, image_col,
                preprocess=bake,
                target_radius=args.target_radius,
                ben_graham=not args.no_ben_graham
            )
        if len(test_df) > 0:
            copy_images(
                test_df, data_dir, output_dir / 'test', image_col,
                preprocess=bake,
                target_radius=args.target_radius,
                ben_graham=not args.no_ben_graham
            )
    else:
        print("\nSkipping image copy (use --copy-images or --preprocess-images)")
        print("Images will be loaded from their original location during training")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")

    if args.folds >= 2:
        image_note = output_dir / 'images' if want_images else data_dir
        print(f"1. Point data.train_dir and data.val_dir at: {image_note}")
        print(f"2. Point data.train_labels/val_labels at "
              f"{output_dir}/train_labels_fold0.csv and val_labels_fold0.csv,")
        print(f"   then repeat for folds 1..{args.folds - 1} and average the results")
    elif want_images:
        print(f"1. Images are in: {output_dir}/train, {output_dir}/val")
        print(f"2. Labels are in: {output_dir}/train_labels.csv, "
              f"{output_dir}/val_labels.csv")
    else:
        print("1. Update the training config to point to:")
        print(f"   - Data dir: {data_dir}")
        print(f"   - Labels: {output_dir}/train_labels.csv, "
              f"{output_dir}/val_labels.csv")

    if bake:
        print("3. Preprocessing is baked into these images -- set "
              "data.preprocess.enabled: false in the config")

    print("4. Run training: uv run dr-train --config config.yaml")
    print()


if __name__ == '__main__':
    main()
