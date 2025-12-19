"""
Data Loading and Preprocessing Utilities
"""

import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


import scipy.io
import re

def load_skeleton_data(data_path, file_pattern="*.skeleton", max_samples=None):
    """
    Load skeleton data from files.
    
    Args:
        data_path: Path to skeleton data directory
        file_pattern: File pattern to match (e.g., "*.skeleton" or "*.mat")
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        sequences: List of skeleton sequences
        labels: List of labels
        metadata: Dictionary with metadata
    """
    sequences = []
    labels = []
    filenames = []
    
    # Find all skeleton files
    # If file_pattern is default but we want to be smart:
    if file_pattern == "*.skeleton" and not glob(os.path.join(data_path, "**", "*.skeleton"), recursive=True):
        # Try .mat if default not found
        if glob(os.path.join(data_path, "**", "*.mat"), recursive=True):
            print("No .skeleton files found, looking for .mat files...")
            file_pattern = "*.mat"
            
    skeleton_files = glob(os.path.join(data_path, "**", file_pattern), recursive=True)
    
    if max_samples is not None:
        skeleton_files = skeleton_files[:max_samples]
    
    print(f"Found {len(skeleton_files)} skeleton files matching {file_pattern}")
    
    for filepath in skeleton_files:
        try:
            # Parse skeleton file (auto-detect format)
            sequence, label = parse_skeleton_file(filepath)
            
            if sequence is not None and len(sequence) > 0:
                sequences.append(sequence)
                labels.append(label)
                filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    if not labels:
        print("Warning: No valid data loaded!")
        return [], [], {'num_samples': 0, 'num_classes': 0}

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    metadata = {
        'num_samples': len(sequences),
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_,
        'label_encoder': label_encoder,
        'filenames': filenames
    }
    
    return sequences, labels_encoded, metadata


def parse_ntu_skeleton_file(filepath):
    """
    Parse NTU RGB+D skeleton file format.
    
    Format:
    - First line: number of frames
    - For each frame:
      - Number of bodies
      - For each body:
        - Body info line (skip)
        - Number of joints (usually 25)
        - For each joint: x, y, z, depth_x, depth_y, rgb_x, rgb_y, quaternion...
    
    Args:
        filepath: Path to .skeleton file
        
    Returns:
        sequence: Skeleton sequence as numpy array (T, 3N) where N=25 joints
        label: Action label (0-59 for NTU RGB+D)
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Extract action label from filename: SxxxCxxxPxxxRxxxAxxx.skeleton
        filename = os.path.basename(filepath)
        if 'A' in filename:
            action_part = filename.split('A')[1].split('.')[0]
            label = int(action_part) - 1  # Convert A001 -> 0, A002 -> 1, etc.
        else:
            label = 0
        
        # Read number of frames
        nframe = int(lines[0].strip())
        
        # Extract skeleton data (use first body only)
        sequence = []
        cursor = 1
        
        for frame in range(nframe):
            if cursor >= len(lines):
                break
            
            # Read body count
            try:
                bodycount = int(lines[cursor].strip())
            except (ValueError, IndexError):
                # Skip this frame if we can't parse body count
                if len(sequence) > 0:
                    sequence.append(sequence[-1])
                else:
                    sequence.append(np.zeros(75))
                cursor += 1
                continue
            
            cursor += 1
            
            if bodycount == 0:
                # Empty frame - use previous frame or zeros
                if len(sequence) > 0:
                    sequence.append(sequence[-1])  # Repeat last frame
                else:
                    # First frame is empty - use zeros
                    sequence.append(np.zeros(75))  # 25 joints * 3 coords
                continue
            
            # Process first body only
            frame_joints = []
            
            for body_idx in range(bodycount):
                if cursor >= len(lines):
                    break
                
                # Skip body info line (contains tracking info, not a number)
                cursor += 1
                if cursor >= len(lines):
                    break
                
                # Read number of joints
                try:
                    njoints = int(lines[cursor].strip())
                except (ValueError, IndexError):
                    # If we can't parse, skip this body
                    break
                
                cursor += 1
                
                # Read joints for this body (only process first body)
                if body_idx == 0:
                    for joint in range(njoints):
                        if cursor >= len(lines):
                            break
                        jointinfo = lines[cursor].strip().split()
                        cursor += 1
                        
                        if len(jointinfo) >= 3:
                            # Extract x, y, z coordinates (first 3 values)
                            try:
                                x, y, z = float(jointinfo[0]), float(jointinfo[1]), float(jointinfo[2])
                                frame_joints.extend([x, y, z])
                            except (ValueError, IndexError):
                                # Skip invalid joint, pad with zeros
                                frame_joints.extend([0.0, 0.0, 0.0])
                else:
                    # Skip joints for other bodies
                    for _ in range(njoints):
                        if cursor < len(lines):
                            cursor += 1
            
            # Ensure we have exactly 75 values (25 joints * 3 coords)
            if len(frame_joints) == 75:
                sequence.append(frame_joints)
            elif len(frame_joints) > 0:
                # Pad or truncate to 75
                while len(frame_joints) < 75:
                    frame_joints.extend([0.0, 0.0, 0.0])
                sequence.append(frame_joints[:75])
            elif len(sequence) > 0:
                # Incomplete frame - use previous frame
                sequence.append(sequence[-1])
            else:
                # First frame incomplete - use zeros
                sequence.append(np.zeros(75))
        
        if len(sequence) == 0:
            return None, None
        
        sequence = np.array(sequence, dtype=np.float32)
        
        return sequence, label
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None


def parse_czu_mat_file(filepath):
    """
    Parse CZU-MHAD .mat file.
    
    Args:
        filepath: Path to .mat file
        
    Returns:
        sequence: Skeleton sequence (T, 66) or similar
        label: Action label
    """
    try:
        data = scipy.io.loadmat(filepath)
        if 'skeleton' in data:
            sequence = data['skeleton']
        elif 'skel' in data:
             sequence = data['skel']
        else:
            # Fallback: look for any large array
            found = False
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] > 10:
                    sequence = v
                    found = True
                    break
            if not found:
                 print(f"Could not find skeleton data in {filepath}")
                 return None, None

        # Ensure correct type
        sequence = sequence.astype(np.float32)
        
        # Extract label from filename: cx_a10_t1.mat -> a10 -> 9
        filename = os.path.basename(filepath)
        match = re.search(r'a(\d+)', filename)
        if match:
            # Convert 1-based to 0-based
            label = int(match.group(1)) - 1
        else:
            label = 0
            
        return sequence, label
        
    except Exception as e:
        print(f"Error parsing .mat file {filepath}: {e}")
        return None, None


def parse_skeleton_file(filepath):
    """
    Parse a skeleton file (adapt based on dataset format).
    
    This function automatically detects NTU RGB+D or CZU/Matlab format.
    
    Args:
        filepath: Path to skeleton file
        
    Returns:
        sequence: Skeleton sequence as numpy array (T, 3N)
        label: Action label
    """
    # Check if it's NTU RGB+D format (has .skeleton extension)
    filename = os.path.basename(filepath)
    if filename.endswith('.skeleton') and 'A' in filename:
        return parse_ntu_skeleton_file(filepath)
    
    # Check if it's Matlab format
    if filename.endswith('.mat'):
        return parse_czu_mat_file(filepath)
    
    # Generic parser for other text formats
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        label = extract_label_from_filename(filepath)
        
        sequence = []
        frame_data = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.isdigit():
                frame_count = int(line)
                frame_data = []
                
                for j in range(frame_count):
                    if i + 1 + j < len(lines):
                        joint_line = lines[i + 1 + j].strip().split()
                        if len(joint_line) >= 3:
                            x, y, z = map(float, joint_line[:3])
                            frame_data.extend([x, y, z])
                
                if len(frame_data) > 0:
                    sequence.append(frame_data)
                
                i += frame_count + 1
            else:
                i += 1
        
        if len(sequence) == 0:
            return None, None
        
        sequence = np.array(sequence)
        
        return sequence, label
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None


def extract_label_from_filename(filepath):
    """
    Extract action label from NTU RGB+D filename.
    
    Format: S001C001P001R001A001.skeleton
    - S001: Setup number
    - C001: Camera ID
    - P001: Person ID
    - R001: Repetition number
    - A001: Action class (this is what we need)
    
    Args:
        filepath: Path to skeleton file
        
    Returns:
        label: Action label (A001, A002, etc.)
    """
    filename = os.path.basename(filepath)
    
    # NTU RGB+D format: S001C001P001R001A001.skeleton
    # Extract action class (A followed by 3 digits)
    if 'A' in filename:
        # Find position of 'A'
        a_pos = filename.find('A')
        if a_pos != -1 and a_pos + 3 < len(filename):
            try:
                # Try to parse Axxx as integer
                return int(filename[a_pos+1:a_pos+4]) - 1
            except ValueError:
                pass
    
    # Try regex for a(\d+) style
    match = re.search(r'a(\d+)', filename)
    if match:
        return int(match.group(1)) - 1
        
    # Fallback: use filename without extension
    return 0


def normalize_sequences(sequences, method='standard'):
    """
    Normalize skeleton sequences.
    
    Args:
        sequences: List of sequences
        method: Normalization method ('standard', 'minmax', 'none')
        
    Returns:
        normalized_sequences: List of normalized sequences
        scaler: Fitted scaler (if applicable)
    """
    if method == 'none':
        return sequences, None
        
    # Check if sequences have consistent dimensions (feature count)
    if not sequences:
        return sequences, None
        
    input_dim = sequences[0].shape[1]
    
    # Flatten for fitting
    # Only use sequences that match the expected dimension
    valid_sequences = [s for s in sequences if s.shape[1] == input_dim]
    if len(valid_sequences) < len(sequences):
         print(f"Warning: dropped {len(sequences) - len(valid_sequences)} sequences with inconsistent dimensions")
    
    if not valid_sequences:
        print("Error: No valid sequences to normalize")
        return sequences, None
        
    all_data = np.concatenate(valid_sequences, axis=0)
    
    if method == 'standard':
        scaler = StandardScaler()
        scaler.fit(all_data)
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(all_data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Transform sequences
    normalized_sequences = []
    for seq in sequences:
        if seq.shape[1] == input_dim:
            normalized_seq = scaler.transform(seq)
            normalized_sequences.append(normalized_seq)
    
    return normalized_sequences, scaler


def prepare_data_for_training(sequences, labels, test_size=0.2, random_state=42):
    """
    Prepare data for training with proper train/test split.
    
    Args:
        sequences: List of sequences
        labels: List of labels
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Filter out empty or inconsistent data
    if len(sequences) != len(labels):
        min_len = min(len(sequences), len(labels))
        sequences = sequences[:min_len]
        labels = labels[:min_len]
        
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
    except ValueError as e:
        print(f"Warning during split: {e}. Falling back to non-stratified split.")
        # Fallback if stratification fails (e.g. single sample for a class)
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            test_size=test_size,
            random_state=random_state
        )
    
    return X_train, X_test, y_train, y_test


def compute_sequence_lengths(sequences):
    """
    Compute sequence length statistics.
    
    Args:
        sequences: List of sequences
        
    Returns:
        stats: Dictionary with length statistics
    """
    if not sequences:
        return {}
        
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        'min': np.min(lengths),
        'max': np.max(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }
    
    return stats


def create_dummy_data(num_samples=100, num_classes=10, seq_length_range=(50, 200), 
                      num_joints=25, random_seed=42):
    """
    Create dummy skeleton data for testing.
    
    Args:
        num_samples: Number of samples
        num_classes: Number of action classes
        seq_length_range: (min, max) sequence length
        num_joints: Number of joints
        random_seed: Random seed
        
    Returns:
        sequences: List of sequences
        labels: List of labels
        metadata: Metadata dictionary
    """
    np.random.seed(random_seed)
    
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # Random sequence length
        seq_len = np.random.randint(seq_length_range[0], seq_length_range[1] + 1)
        
        # Random skeleton sequence (T, 3N)
        sequence = np.random.randn(seq_len, 3 * num_joints)
        
        # Add some structure based on label
        label = i % num_classes
        sequence += 0.1 * np.sin(np.linspace(0, 2 * np.pi * (label + 1), seq_len))[:, None]
        
        sequences.append(sequence)
        labels.append(label)
    
    metadata = {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'class_names': [f'Action_{i}' for i in range(num_classes)],
        'num_joints': num_joints
    }
    
    return sequences, np.array(labels), metadata


