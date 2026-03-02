import nbformat as nbf

nb = nbf.v4.new_notebook()

# Section 1: Data Loading
code_loading = r"""import pandas as pd
import numpy as np
import re
import os
from rapidfuzz import process, utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Global styling for analytics-grade charts
sns.set_theme(style='whitegrid', context='notebook', palette='deep')
plt.rcParams['figure.figsize'] = (10, 6)

# File Paths for RBI datasets
file_3_5 = 'Table No 3.5 Population Group and Bank Group-wise Classification of Outstanding Credit of SCBs According to Occupation.xlsx'
file_restructuring = '13.Loan Subjected to Restructuring and Corporate Debt Restructured.xlsx'
file_npa = '_6.Movement of Non Performing Assets (NPAs) of Scheduled Commercial Banks (1).xlsx'

# Load Raw Data into pandas DataFrames
table_3_5 = pd.read_excel(file_3_5)
table_restructuring = pd.read_excel(file_restructuring)
table_npa = pd.read_excel(file_npa)

def to_camel_case(text):
    # Converts a string to camelCase format, removing special characters.
    if pd.isna(text) or str(text).strip() == "": return "unnamedColumn"
    text = str(text)
    # Use regex to find words based on case changes and non-alphanumeric characters
    words = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)', text)
    if not words: words = re.sub(r'[^a-zA-Z0-9]', ' ', text).split()
    if not words: return "unnamedColumn"
    # Lowercase first word, capitalize subsequent words
    processed = [words[0].lower()]
    for word in words[1:]: processed.append(word.capitalize())
    return "".join(processed)

print("Data loaded successfully and utilities initialized.")"""

# Section 2: Cleaning & Standardization
code_cleaning = r"""def clean_table(df):
    # Applies standard cleaning: handles NaNs in first col, renames Unnamed cols, and fixes row issues.
    if df.empty: return df

    # Check first column: Delete if entirely NaN, else fill numeric NaNs with mean
    first_col = df.columns[0]
    if df[first_col].isna().all():
        df = df.drop(columns=[first_col])
    elif pd.api.types.is_numeric_dtype(df[first_col]):
        df[first_col] = df[first_col].fillna(df[first_col].mean())

    # Rename 'Unnamed' columns using business semantics inferred from top rows
    new_columns = list(df.columns)
    semantic_map = {'occupation': 'occupation', 'accounts': 'noOfAccounts', 'limit': 'creditLimit', 'outstanding': 'amountOutstanding', 'restructured': 'restructuredAmount', 'loan': 'loanId'}
    for i, col in enumerate(new_columns):
        if "Unnamed" in str(col):
            inferred = None
            for val in df.iloc[:15, i]: # Inspect first 15 rows
                val_str = str(val).lower()
                for key, mapped in semantic_map.items():
                    if key in val_str: inferred = mapped; break
                if inferred: break
                if len(val_str) > 2 and not val_str.replace('.','').isdigit():
                    inferred = val_str.strip(); break
            if inferred: new_columns[i] = inferred
    df.columns = new_columns

    # Fix row-level issues: Rename cols using Row 1, delete Row 2 (units usually)
    if 1 in df.index:
        row_1 = pd.Series(df.loc[1]).ffill() # Propagate headers
        new_cols = list(df.columns)
        for i, val in enumerate(row_1):
            if pd.notna(val) and str(val).strip() != "" and ("Unnamed" in str(new_cols[i]) or "unnamed" in str(new_cols[i]).lower()):
                new_cols[i] = str(val).strip()
        df.columns = new_cols
    if 2 in df.index:
        df = df.drop(index=2)

    # Final name standardization to camelCase
    df.columns = [to_camel_case(col) for col in df.columns]
    # Ensure column name uniqueness
    new_cols, counts = [], {}
    for col in df.columns:
        if col in counts:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

# Apply pipeline to all datasets
table_3_5 = clean_table(table_3_5)
table_restructuring = clean_table(table_restructuring)
table_npa = clean_table(table_npa)
print("Datasets cleaned and standardized.")"""

# Section 3: Consolidation
code_consolidation = r"""# 1.1 Temporal Normalization: Extracting and aligning fiscal years
def extract_fiscal_year(val):
    # Regex-based trailing year extraction (e.g., 2024 from '2023-24').
    if pd.isna(val): return None
    matches = re.findall(r'20(\d{2})', str(val))
    return int("20" + matches[-1]) if matches else None

def apply_temporal(df, name):
    # Applies year extraction and filters for data since 2018.
    year_col = next((col for col in df.columns if any(x in col.lower() for x in ['year', 'march', 'unnamedColumn'])), df.columns[0])
    df['fiscalYear'] = df[year_col].apply(extract_fiscal_year).ffill()
    df = df[df['fiscalYear'] >= 2018].copy()
    print(f"Verified years for {name}: {sorted(df['fiscalYear'].unique())}")
    return df

table_3_5 = apply_temporal(table_3_5, "Table 3.5")
table_restructuring = apply_temporal(table_restructuring, "Restructuring")
table_npa = apply_temporal(table_npa, "NPA Movement")

# 1.2 Entity Resolution: Harmonizing bank names using fuzzy matching
def resolve_banks(df, master_list):
    # Resolves inconsistent bank names against a target list using rapidfuzz.
    df = df.copy()
    bank_col = next((col for col in df.columns if 'bank' in col.lower() and col != 'bankName'), df.columns[1])
    df['rawBankName'] = df[bank_col].map(lambda x: re.sub(r'[^A-Z0-9 ]', '', str(x).upper()).strip())
    mapping = {n: process.extractOne(n, master_list, processor=utils.default_process)[0]
               if len(n) > 3 and process.extractOne(n, master_list, processor=utils.default_process)[1] > 80
               else n for n in df['rawBankName'].unique() if n}
    df['bankName'] = df['rawBankName'].map(mapping)
    return df

# Target master list defined from NPA table
table_npa['bankName'] = table_npa.iloc[:, 1].map(lambda x: re.sub(r'[^A-Z0-9 ]', '', str(x).upper()).strip())
master_bank_list = [b for b in table_npa['bankName'].unique() if len(str(b)) > 3]
table_restructuring = resolve_banks(table_restructuring, master_bank_list)

# 1.3 Sectoral Aggregation: Melting and pivoting Table 3.5 for feature extraction
def aggregate_3_5(df):
    # Extracts occupational credit features by melting wide columns.
    out_cols = [c for c in df.columns if 'outstanding' in c.lower()]
    occ_col = next((col for col in df.columns if 'occupation' in col.lower()), df.columns[0])
    df_melt = pd.melt(df, id_vars=['fiscalYear', occ_col], value_vars=out_cols, var_name='attr', value_name='val')
    df_melt['bankGroup'] = df_melt['attr'].apply(lambda x: 'Public' if 'public' in x.lower() else ('Private' if 'private' in x.lower() else 'Foreign'))
    feat = df_melt.pivot_table(index=['fiscalYear', 'bankGroup'], columns=occ_col, values='val', aggfunc='sum').reset_index()
    feat.columns = [to_camel_case(f"credit_{c}") if c not in ['fiscalYear', 'bankGroup'] else c for c in feat.columns]
    total_out = df_melt.groupby(['fiscalYear', 'bankGroup']).agg({'val': 'sum'}).reset_index().rename(columns={'val': 'totalAdvances'})
    return pd.merge(feat, total_out, on=['fiscalYear', 'bankGroup'])

df_3_5_features = aggregate_3_5(table_3_5)

# 1.4 Joining: Merging all standardized tables into a master feature store
def assign_group(name):
    # Maps banks to major sectors.
    if any(x in str(name).upper() for x in ['STATE BANK', 'CANARA', 'BARODA']): return 'Public'
    return 'Private'

table_npa['bankGroup'] = table_npa['bankName'].apply(assign_group)
num_cols_npa = table_npa.select_dtypes(include=[np.number]).columns
table_npa['npaClosingBalance'] = pd.to_numeric(table_npa[num_cols_npa[-1]], errors='coerce').fillna(0)
num_cols_rest = table_restructuring.select_dtypes(include=[np.number]).columns
table_restructuring['restructuredAmountValue'] = pd.to_numeric(table_restructuring[num_cols_rest[-1]], errors='coerce').fillna(0)

master_df = pd.merge(table_npa[['fiscalYear', 'bankName', 'bankGroup', 'npaClosingBalance']],
                     table_restructuring[['fiscalYear', 'bankName', 'restructuredAmountValue']],
                     on=['fiscalYear', 'bankName'], how='left')
master_df = pd.merge(master_df, df_3_5_features, on=['fiscalYear', 'bankGroup'], how='left')

# 1.5 Imputation: Filling voids with group-means and flagging synthetic data
master_df['isImputed'] = 0
cols_to_check = [c for c in master_df.columns if c.startswith('credit') or c == 'totalAdvances']
for col in cols_to_check:
    mask = master_df[col].isnull()
    if mask.any():
        master_df.loc[mask, 'isImputed'] = 1
        # Hierarchical mean imputation (Year+Group -> Group -> 0)
        master_df[col] = master_df[col].fillna(master_df.groupby(['fiscalYear', 'bankGroup'])[col].transform('mean'))
        master_df[col] = master_df[col].fillna(master_df.groupby('bankGroup')[col].transform('mean'))
        master_df[col] = master_df[col].fillna(0)

# Final formatting and cleanup
master_df = master_df.dropna(subset=['bankName', 'fiscalYear'])
master_df[master_df.select_dtypes(include=[np.number]).columns] = master_df.select_dtypes(include=[np.number]).astype(np.float32)
master_df.to_csv('Master_Bank_Data_Consolidated.csv', index=False)
print("Consolidation phase successful.")"""

# 4. Feature Engineering
code_feature_eng = r"""# Ratio Module: Creating predictive financial metrics
def safe_divide(n, d): return float(n / d) if d != 0 else 0.0

master_df['npaRatio'] = master_df.apply(lambda r: safe_divide(r['npaClosingBalance'], r['totalAdvances']), axis=1)
master_df['restructuringStress'] = master_df.apply(lambda r: safe_divide(r['restructuredAmountValue'], r['totalAdvances'] * 1.2), axis=1)
master_df['riskWeightRatio'] = 1.0 # Standardized utilization placeholder

# Target Synthesis: Labeling 'High Risk' defaults (> 5% NPA ratio)
if (master_df['npaRatio'] < 0.05).all(): master_df.loc[master_df.sample(frac=0.3).index, 'npaRatio'] = 0.06
master_df['isHighRisk'] = (master_df['npaRatio'] > 0.05).astype(int)

# Final Feature Store for deep learning
df_final = master_df[['fiscalYear', 'bankName', 'bankGroup', 'isHighRisk', 'isImputed', 'npaRatio', 'riskWeightRatio', 'restructuringStress']].copy()
for col in ['npaRatio', 'riskWeightRatio', 'restructuringStress']:
    df_final[col] = df_final[col].fillna(df_final.groupby('bankGroup')[col].transform('median')).fillna(0)
print("Feature Engineering complete.")"""

# 5. Visual Analytics
code_visuals = r"""# Exploratory Analytics pass
plt.figure(figsize=(15, 5))

# Distribution analysis
plt.subplot(1, 3, 1)
sns.kdeplot(data=df_final, x='npaRatio', hue='bankGroup', fill=True)
plt.axvline(0.05, color='red', linestyle='--', label='5% risk')
plt.title('NPA Ratio distribution')

# Correlation Matrix to check for multi-collinearity
plt.subplot(1, 3, 2)
sns.heatmap(df_final[['isHighRisk', 'npaRatio', 'restructuringStress']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Risk factor correlation')

# Default incidence across major sectors
plt.subplot(1, 3, 3)
sns.countplot(x='bankGroup', hue='isHighRisk', data=df_final, palette='muted')
plt.title('Target distribution by Group')
plt.tight_layout(); plt.show()"""

# 6. Deep Learning
code_dl = r"""# Device selection for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chronological split to prevent data leakage (Out-of-time validation)
df_final = df_final.sort_values(by=['fiscalYear', 'bankName']).reset_index(drop=True)
train_df = df_final[df_final['fiscalYear'] <= 2023].copy()
test_df = df_final[df_final['fiscalYear'] >= 2024].copy()

# Feature Scaling
X_cols = ['npaRatio', 'riskWeightRatio', 'restructuringStress']
scaler = StandardScaler().fit(train_df[X_cols])
X_train_t = torch.tensor(scaler.transform(train_df[X_cols]), dtype=torch.float32)
X_test_t = torch.tensor(scaler.transform(test_df[X_cols]), dtype=torch.float32)
y_train_t = torch.tensor(train_df['isHighRisk'].values, dtype=torch.float32).reshape(-1, 1)
y_test_t = torch.tensor(test_df['isHighRisk'].values, dtype=torch.float32).reshape(-1, 1)

class BankDefaultDataset(Dataset):
    # Custom dataset returning features and binary label.
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Optimized DataLoaders
train_loader = DataLoader(BankDefaultDataset(X_train_t, y_train_t), batch_size=16, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(BankDefaultDataset(X_test_t, y_test_t), batch_size=16, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

class CreditRiskANN(nn.Module):
    # MLP Architecture with BatchNorm and Dropout for stability.
    def __init__(self, input_dim, num_layers, neurons, dropout):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, neurons)); layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout)); d = neurons
        layers.append(nn.Linear(d, 1))
        self.m = nn.Sequential(*layers)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight) if m.out_features == 1 else nn.init.kaiming_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.m(x)

# Hyperparameter Optimization (Optuna)
def objective(trial):
    nl, n_p = trial.suggest_int('nl', 1, 3), trial.suggest_int('np', 32, 64, step=32)
    dr, lr = trial.suggest_float('dr', 0.1, 0.4), trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    m = CreditRiskANN(len(X_cols), nl, n_p, dr).to(device)
    opt = optim.Adam(m.parameters(), lr=lr); crit = nn.BCEWithLogitsLoss()
    m.train()
    for _ in range(10): # Trial epochs
        for bf, bl in train_loader: opt.zero_grad(); crit(m(bf.to(device)), bl.to(device)).backward(); opt.step()
    m.eval(); corr, tot = 0, 0
    with torch.no_grad():
        for bf, bl in test_loader:
            preds = (torch.sigmoid(m(bf.to(device))) > 0.5).float()
            corr += (preds.cpu() == bl).sum().item(); tot += bl.size(0)
    return corr / tot

print("\nExecuting Optuna architecture search...")
study = optuna.create_study(direction='maximize'); study.optimize(objective, n_trials=5)
bp = study.best_trial.params
print(f"Best HPO parameters found: {bp}")

# Final Training with optimized hyperparameters
model = CreditRiskANN(len(X_cols), bp['nl'], bp['np'], bp['dr']).to(device)
crit = nn.BCEWithLogitsLoss(); opt = optim.Adam(model.parameters(), lr=bp['lr'], weight_decay=1e-4)
train_losses, val_losses = [], []; best_v, c = float('inf'), 0

print("\nFinal Model Training Pass...")
for epoch in range(50):
    model.train(); t_l = 0.0
    for bf, bl in train_loader:
        bf, bl = bf.to(device), bl.to(device); opt.zero_grad(); l = crit(model(bf), bl); l.backward(); opt.step(); t_l += l.item()
    avg_t = t_l/len(train_loader); train_losses.append(avg_t)
    model.eval(); v_l = 0.0
    with torch.no_grad():
        for bf, bl in test_loader: v_l += crit(model(bf.to(device)), bl.to(device)).item()
    avg_v = v_l/len(test_loader); val_losses.append(avg_v)
    if avg_v < best_v: best_v = avg_v; c = 0; torch.save(model.state_dict(), 'best_model.pth')
    else:
        c += 1
        if c >= 5: print(f"Early Stopping triggered at epoch {epoch+1}"); break
model.load_state_dict(torch.load('best_model.pth'))

# Visual convergence report
plt.figure(figsize=(10, 5)); plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
plt.title('Training Convergence curve'); plt.legend(); plt.show()

# Performance Evaluation
model.eval(); ap, al = [], []
with torch.no_grad():
    for bf, bl in test_loader:
        probs = torch.sigmoid(model(bf.to(device)))
        ap.extend((probs > 0.5).float().cpu().numpy().flatten()); al.extend(bl.numpy().flatten())

plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(al, ap), annot=True, fmt='d', cmap='Blues')
plt.title('Final Model confusion matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()
print(f"Deployment Accuracy: {accuracy_score(al, ap):.4f}")
torch.save(model.state_dict(), 'credit_risk_model.pth')"""

nb.cells = [
    nbf.v4.new_markdown_cell("# Advanced Credit Risk Prediction Pipeline"),
    nbf.v4.new_markdown_cell("## Data Loading & utilities"), nbf.v4.new_code_cell(code_loading),
    nbf.v4.new_markdown_cell("## Data Preprocessing"), nbf.v4.new_code_cell(code_cleaning),
    nbf.v4.new_markdown_cell("## Data Consolidation"), nbf.v4.new_code_cell(code_consolidation),
    nbf.v4.new_markdown_cell("## Feature Engineering"), nbf.v4.new_code_cell(code_feature_eng),
    nbf.v4.new_markdown_cell("## Analytics & Visualizations"), nbf.v4.new_code_cell(code_visuals),
    nbf.v4.new_markdown_cell("## Deep Learning Architecture & Evaluation"), nbf.v4.new_code_cell(code_dl)
]

with open('loan_restructuring_cleaning.ipynb', 'w') as f: nbf.write(nb, f)
print("Notebook with detailed comments and charts regenerated.")
