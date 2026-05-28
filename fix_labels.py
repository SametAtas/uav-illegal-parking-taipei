import os
import glob

def fix_labels(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    count = 0
    for txt in txt_files:
        with open(txt, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Change class index from 1 to 0 (or whatever it is to 0)
            parts[0] = '0'
            new_lines.append(' '.join(parts) + '\n')
            
        with open(txt, 'w') as f:
            f.writelines(new_lines)
        count += 1
    print(f"Fixed {count} label files in {folder_path}")

base_dir = "CARPK-1/extracted/CarPK/CarPK"
fix_labels(os.path.join(base_dir, "train", "labels"))
fix_labels(os.path.join(base_dir, "test", "labels"))
