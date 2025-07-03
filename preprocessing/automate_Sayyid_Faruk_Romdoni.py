import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os 

def preprocess(df_raw, output_path):
    print("Memulai proses preprocessing...")

    df_processed = df_raw.copy()
    
    df_processed.drop_duplicates(inplace=True)
    
    imputer = SimpleImputer(strategy='median')
    numerical_cols_with_missing = ['trestbps', 'chol', 'thalach']
    df_processed[numerical_cols_with_missing] = imputer.fit_transform(df_processed[numerical_cols_with_missing])    
    
    df_processed['sex'] = df_processed['sex'].map({'male': 1, 'female': 0})
    df_processed['target'] = df_processed['target'].map({'yes': 1, 'no': 0})
    
    categorical_to_encode = ['cp', 'restecg', 'slope', 'thal']
    df_processed = pd.get_dummies(df_processed, columns=categorical_to_encode, drop_first=True, dtype=int)
    
    numerical_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    scaler = StandardScaler()
    df_processed[numerical_to_scale] = scaler.fit_transform(df_processed[numerical_to_scale])
        
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)        
        df_processed.to_csv(output_path, index=False)
        print(f"Preprocessing selesai. Data bersih disimpan di: {output_path}")
    except Exception as e:
        print(f"Gagal menyimpan file. Error: {e}")
        
    return df_processed

if __name__ == '__main__':
    input_file = './heart disease_raw.csv'
    output_file = 'preprocessing/heart_disease_preprocessed.csv'
    
    try:
        raw_data = pd.read_csv(input_file, index_col=0)
        
        cleaned_data = preprocess(raw_data, output_file)
        
        print("\nVerifikasi 5 baris pertama data yang disimpan:")
        print(cleaned_data.head())
        
    except FileNotFoundError:
        print(f"Error: File input '{input_file}' tidak ditemukan.")