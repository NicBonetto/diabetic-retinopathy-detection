import argparse
import shutil
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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

def copy_images(
    df: pd.DataFrame,
    source_dir: Path,
    dest_dir: Path,
    image_col: str
) -> None:
    """Copy images from source to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f'Copying {len(df)} images to {dest_dir}')

    copied = 0

    for image_id in tqdm(df[image_col], desc='Copying images'):
        source_file = source_dir / f'{image_id}.png'
        dest_file = dest_dir / source_file.name
        shutil.copy2(source_file, dest_file)
        copied += 1

    print(f'Copied: {copied}')

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
    
    args = parser.parse_args()
    
    # Convert to Path objects
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
    
    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(args.labels)
    print(f"Loaded {len(labels_df)} samples")
    
    # Detect column names
    image_col = 'id_code'
    label_col = 'diagnosis'
    print(f"Using columns: image_id='{image_col}', label='{label_col}'")
    
    # Create splits
    print("\nCreating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_split(
        labels_df,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Print statistics
    print_split_statistics(train_df, val_df, test_df, label_col)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label files with standardized column names
    print("Saving label files...")
    
    # Standardize to 'image_id' and 'diagnosis'
    train_df_out = train_df.copy()
    train_df_out = train_df_out.rename(columns={image_col: 'image_id', label_col: 'diagnosis'})
    train_df_out.to_csv(output_dir / 'train_labels.csv', index=False)
    print(f"  Saved: {output_dir / 'train_labels.csv'}")
    
    val_df_out = val_df.copy()
    val_df_out = val_df_out.rename(columns={image_col: 'image_id', label_col: 'diagnosis'})
    val_df_out.to_csv(output_dir / 'val_labels.csv', index=False)
    print(f"  Saved: {output_dir / 'val_labels.csv'}")
    
    if len(test_df) > 0:
        test_df_out = test_df.copy()
        test_df_out = test_df_out.rename(columns={image_col: 'image_id', label_col: 'diagnosis'})
        test_df_out.to_csv(output_dir / 'test_labels.csv', index=False)
        print(f"  Saved: {output_dir / 'test_labels.csv'}")
    
    # Optionally copy images
    if args.copy_images:
        print("\nCopying images to output directory...")
        
        copy_images(train_df, data_dir, output_dir / 'train', image_col)
        copy_images(val_df, data_dir, output_dir / 'val', image_col)
        
        if len(test_df) > 0:
            copy_images(test_df, data_dir, output_dir / 'test', image_col)
    else:
        print("\nSkipping image copy (use --copy-images to enable)")
        print("Images will be loaded from original location during training")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    if args.copy_images:
        print(f"1. Images are in: {output_dir}/train, {output_dir}/val")
        print(f"2. Labels are in: {output_dir}/train_labels.csv, {output_dir}/val_labels.csv")
    else:
        print(f"1. Update training config to point to:")
        print(f"   - Data dir: {data_dir}")
        print(f"   - Labels: {output_dir}/train_labels.csv, {output_dir}/val_labels.csv")
    print("3. Run training: python scripts/train.py --config configs/train_resnet50.yaml")
    print()


if __name__ == '__main__':
    main()
