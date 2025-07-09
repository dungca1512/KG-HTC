import pandas as pd
import os
import json
from pathlib import Path

def analyze_wos_structure():
    """Phân tích cấu trúc WoS dataset chuẩn"""
    
    print("=== WoS DATASET STANDARD STRUCTURE ===")
    print("""
    Standard WoS structure should be:
    
    dataset/wos/
    ├── Meta-data/
    │   └── Data.xlsx           # Main dataset file
    ├── wos_total.txt           # All text data concatenated
    ├── train_texts.txt         # Training texts (one per line)
    ├── train_labels.txt        # Training labels (one per line)
    ├── test_texts.txt          # Test texts (one per line)
    ├── test_labels.txt         # Test labels (one per line)
    └── val_texts.txt           # Validation texts (one per line)
    └── val_labels.txt          # Validation labels (one per line)
    
    Data.xlsx columns:
    - Y1: Level 1 label index (0-6 for 7 domains)
    - Y2: Level 2 label index (0-142 for 143 areas)  
    - Y: Combined hierarchical label
    - Domain: Level 1 category name (CS, Medical, etc.)
    - area: Level 2 category name (Symbolic computation, etc.)
    - keywords: Research keywords
    - Abstract: Paper abstract text
    """)

def check_current_data_structure(file_path="Data.xlsx"):
    """Kiểm tra cấu trúc dữ liệu hiện tại"""
    
    print("\n=== CHECKING CURRENT DATA STRUCTURE ===")
    
    try:
        df = pd.read_excel(file_path)
        print(f"✓ File loaded successfully: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Kiểm tra các cột bắt buộc
        required_cols = ['Y1', 'Y2', 'Y', 'Domain', 'area', 'Abstract']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
        else:
            print("✓ All required columns present")
        
        # Kiểm tra phân bố dữ liệu
        print(f"\nData distribution:")
        print(f"- Total records: {len(df)}")
        print(f"- Unique Domains (Y1): {df['Y1'].nunique()} (expected: 7)")
        print(f"- Unique Areas (Y2): {df['Y2'].nunique()} (expected: 143)")
        print(f"- Domain names: {sorted(df['Domain'].unique())}")
        print(f"- Sample areas: {sorted(df['area'].unique())[:10]}")
        
        # Kiểm tra label consistency
        domain_mapping = df.groupby('Y1')['Domain'].first().to_dict()
        area_mapping = df.groupby('Y2')['area'].first().to_dict()
        
        print(f"\nLabel mappings:")
        print(f"Domain mapping (Y1 -> Domain): {domain_mapping}")
        print(f"Area count per domain:")
        for domain in sorted(df['Domain'].unique()):
            count = df[df['Domain'] == domain]['area'].nunique()
            print(f"  {domain}: {count} areas")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None

def create_wos_text_files(df, output_dir="dataset/wos", split_ratio=(0.7, 0.15, 0.15)):
    """Tạo các file text theo chuẩn WoS"""
    
    print(f"\n=== CREATING WoS TEXT FILES ===")
    
    if df is None:
        print("No data to process")
        return
    
    # Tạo thư mục
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    n_total = len(df_shuffled)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train+n_val]
    test_df = df_shuffled[n_train+n_val:]
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Tạo các file text
    splits = {
        'train': train_df,
        'val': val_df, 
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        
        # Texts file (abstracts)
        texts_file = f"{output_dir}/{split_name}_texts.txt"
        with open(texts_file, 'w', encoding='utf-8') as f:
            for abstract in split_df['Abstract']:
                # Clean text
                clean_text = str(abstract).replace('\n', ' ').replace('\r', ' ').strip()
                f.write(clean_text + '\n')
        
        # Labels file (hierarchical: domain|area)
        labels_file = f"{output_dir}/{split_name}_labels.txt"
        with open(labels_file, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                domain = str(row['Domain']).strip()
                area = str(row['area']).strip()
                label = f"{domain}|{area}"
                f.write(label + '\n')
        
        print(f"✓ Created {split_name}_texts.txt and {split_name}_labels.txt")
    
    # Tạo file tổng hợp
    total_file = f"{output_dir}/wos_total.txt"
    with open(total_file, 'w', encoding='utf-8') as f:
        for abstract in df_shuffled['Abstract']:
            clean_text = str(abstract).replace('\n', ' ').replace('\r', ' ').strip()
            f.write(clean_text + '\n')
    
    print(f"✓ Created wos_total.txt")
    
    # Tạo metadata file
    metadata = {
        'total_records': len(df),
        'domains': df['Domain'].unique().tolist(),
        'areas': df['area'].unique().tolist(),
        'domain_count': df['Domain'].nunique(),
        'area_count': df['area'].nunique(),
        'split_ratio': split_ratio,
        'split_sizes': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
    }
    
    metadata_file = f"{output_dir}/metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created metadata.json")

def create_label_mappings(df, output_dir="dataset/wos"):
    """Tạo file mapping cho labels"""
    
    print(f"\n=== CREATING LABEL MAPPINGS ===")
    
    # Domain mapping
    domain_to_id = {domain: idx for idx, domain in enumerate(sorted(df['Domain'].unique()))}
    id_to_domain = {idx: domain for domain, idx in domain_to_id.items()}
    
    # Area mapping  
    area_to_id = {area: idx for idx, area in enumerate(sorted(df['area'].unique()))}
    id_to_area = {idx: area for area, idx in area_to_id.items()}
    
    # Save mappings
    mappings = {
        'domain_to_id': domain_to_id,
        'id_to_domain': id_to_domain,
        'area_to_id': area_to_id,
        'id_to_area': id_to_area
    }
    
    mapping_file = f"{output_dir}/label_mappings.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created label_mappings.json")
    print(f"Domain mapping: {domain_to_id}")
    print(f"Total areas: {len(area_to_id)}")

def convert_ai_agent_to_wos_format(input_file="AI Agent_Research_Dữ liệu mẫu.xlsx"):
    """Chuyển đổi AI Agent data sang định dạng WoS chuẩn"""
    
    print("\n=== CONVERTING AI AGENT DATA TO WoS FORMAT ===")
    
    try:
        # Đọc cả 2 sheets
        excel_file = pd.ExcelFile(input_file)
        print(f"Available sheets: {excel_file.sheet_names}")
        
        # Tìm data sheet
        data_sheet = None
        tag_sheet = None
        
        for sheet in excel_file.sheet_names:
            if "raw review" in sheet.lower() or "bản sao" in sheet.lower():
                data_sheet = sheet
            elif "tag" in sheet.lower():
                tag_sheet = sheet
        
        if not data_sheet:
            print("Data sheet not found!")
            return None
        
        # Đọc data
        df_data = pd.read_excel(input_file, sheet_name=data_sheet)
        print(f"Data sheet '{data_sheet}': {df_data.shape}")
        print(f"Columns: {list(df_data.columns)}")
        
        if tag_sheet:
            df_tags = pd.read_excel(input_file, sheet_name=tag_sheet)
            print(f"Tag sheet '{tag_sheet}': {df_tags.shape}")
            print(f"Tag columns: {list(df_tags.columns)}")
        else:
            df_tags = None
        
        # Auto-detect columns
        text_col = None
        label_col = None
        
        for col in df_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['review', 'text', 'content', 'comment']):
                text_col = col
            if any(keyword in col_lower for keyword in ['label', 'tag', 'category', 'sentiment']):
                label_col = col
        
        print(f"Detected text column: {text_col}")
        print(f"Detected label column: {label_col}")
        
        if not text_col or not label_col:
            print("Cannot auto-detect required columns!")
            print("Please manually specify column names")
            return None
        
        # Tạo WoS format
        wos_data = []
        unique_labels = df_data[label_col].unique()
        
        # Tạo hierarchy mapping đơn giản (có thể cải thiện với tag sheet)
        domain_mapping = {}
        area_mapping = {}
        
        for i, label in enumerate(unique_labels):
            if pd.notna(label):
                # Đơn giản hóa: dùng label làm cả domain và area
                domain = str(label).strip()
                area = str(label).strip()
                
                domain_mapping[label] = domain
                area_mapping[label] = area
        
        # Tạo Y1, Y2 mapping
        unique_domains = list(set(domain_mapping.values()))
        unique_areas = list(set(area_mapping.values()))
        
        domain_to_y1 = {domain: i for i, domain in enumerate(sorted(unique_domains))}
        area_to_y2 = {area: i for i, area in enumerate(sorted(unique_areas))}
        
        # Convert data
        for _, row in df_data.iterrows():
            if pd.notna(row[text_col]) and pd.notna(row[label_col]):
                
                domain = domain_mapping[row[label_col]]
                area = area_mapping[row[label_col]]
                
                y1 = domain_to_y1[domain]
                y2 = area_to_y2[area]
                y = y1 * 1000 + y2  # Combined label
                
                wos_data.append({
                    'Y1': y1,
                    'Y2': y2,
                    'Y': y,
                    'Domain': domain,
                    'area': area,
                    'keywords': '',  # Placeholder
                    'Abstract': str(row[text_col]).strip()
                })
        
        # Tạo DataFrame
        wos_df = pd.DataFrame(wos_data)
        
        # Lưu file
        output_dir = "dataset/ai_agent_wos"
        os.makedirs(f"{output_dir}/Meta-data", exist_ok=True)
        
        wos_df.to_excel(f"{output_dir}/Meta-data/Data.xlsx", index=False)
        print(f"✓ Saved WoS format data: {wos_df.shape}")
        
        return wos_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    
    # 1. Hiển thị cấu trúc chuẩn
    analyze_wos_structure()
    
    # 2. Kiểm tra dữ liệu hiện tại
    df = check_current_data_structure("Data.xlsx")
    
    if df is not None:
        # 3. Tạo các file text chuẩn
        create_wos_text_files(df, "dataset/wos")
        
        # 4. Tạo label mappings
        create_label_mappings(df, "dataset/wos")
        
        print("\n=== COMPLETED ===")
        print("WoS dataset structure created successfully!")
        print("Files created:")
        print("- dataset/wos/train_texts.txt & train_labels.txt")
        print("- dataset/wos/val_texts.txt & val_labels.txt")
        print("- dataset/wos/test_texts.txt & test_labels.txt")
        print("- dataset/wos/wos_total.txt")
        print("- dataset/wos/metadata.json")
        print("- dataset/wos/label_mappings.json")
    
    # 5. Nếu có AI Agent data, convert nó
    if os.path.exists("AI Agent_Research_Dữ liệu mẫu.xlsx"):
        print("\n" + "="*50)
        ai_df = convert_ai_agent_to_wos_format("AI Agent_Research_Dữ liệu mẫu.xlsx")
        if ai_df is not None:
            create_wos_text_files(ai_df, "dataset/ai_agent_wos")
            create_label_mappings(ai_df, "dataset/ai_agent_wos")
            
            