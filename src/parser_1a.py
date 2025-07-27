import fitz  
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import json


FEATURE_COLS = [
    'char_count', 'word_count', 'is_all_caps',
    'size_ratio', 'size_rank', 'is_bold', 'is_numbered_list'
]

def extract_features(pdf_path):
    """Extracts the 7 features required by the trained model."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [1A Parser] Error opening PDF {pdf_path}: {e}")
        return pd.DataFrame()

    all_spans_data = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=11).get("blocks", [])
        for b in blocks:
            if b.get('lines'):
                for line in b['lines']:
                    for span in line['spans']:
                        span['page_num'] = page_num
                        span['y0'] = span['bbox'][1]
                        all_spans_data.append(span)

    if not all_spans_data: return pd.DataFrame()
    df = pd.DataFrame(all_spans_data)

    df['char_count'] = df['text'].str.len()
    df['word_count'] = df['text'].str.strip().str.split().str.len()
    df['is_all_caps'] = df['text'].apply(lambda x: x.isupper() and len(x) > 2)

    if 'size' in df.columns and not df.empty:
        p_size = np.median(df['size'])
        df['size_ratio'] = df['size'] / (p_size + 1e-6)
        df['size_rank'] = df['size'].rank(method='dense', ascending=False)
    else:
        df['size'] = 0; df['size_ratio'] = 1.0; df['size_rank'] = 1.0

    df['is_bold'] = df['font'].str.contains("bold", case=False, na=False).astype(int)
    df['is_numbered_list'] = df['text'].str.strip().str.match(r'^\d+(\.\d+)*').fillna(False).astype(int)

    return df

def get_document_structure(pdf_path, model_file_path):
    """Main 1A function using the ML model to get title and outline."""
    print(f"--- [1A] Parsing structure for: {os.path.basename(pdf_path)} ---")

    features_df = extract_features(pdf_path)
    if features_df.empty:
        return {"title": "Extraction Failed", "outline": []}

    model = lgb.Booster(model_file=model_file_path)
    for col in FEATURE_COLS:
        if col not in features_df.columns:
            features_df[col] = 0

    X_predict = features_df[FEATURE_COLS]
    predictions = model.predict(X_predict)

    features_df['pred_prob_title'] = predictions[:, 1]
    features_df['pred_prob_heading'] = predictions[:, 2]

    title = ""
    page0_df = features_df[features_df['page_num'] == 0]
    if not page0_df.empty:
        title_candidates = page0_df[page0_df['pred_prob_title'] > 0.5]
        if not title_candidates.empty:
            best_title_candidate = title_candidates.loc[title_candidates['size_rank'].idxmin()]
            title = best_title_candidate['text'].strip()

    headings_df = features_df[features_df['pred_prob_heading'] > 0.6].copy()

    outline = []
    if not headings_df.empty:
        headings_df = headings_df.sort_values(by=['page_num', 'y0'])
        unique_sizes = sorted(headings_df['size'].unique(), reverse=True)
        size_to_level = {size: f"H{i+1}" for i, size in enumerate(unique_sizes[:3])}

        for _, row in headings_df.iterrows():
            clean_text = row['text'].strip()
            if len(clean_text) > 1 and clean_text.lower() != title.lower():
                level = size_to_level.get(row['size'], 'H3')
                outline.append({
                    "level": level,
                    "text": clean_text,
                    "page": int(row['page_num']),
                    "bbox": row['bbox']
                })

    return {"title": title, "outline": outline}