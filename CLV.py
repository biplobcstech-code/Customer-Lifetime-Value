"""
===============================================================
  Customer Lifetime Value (CLV) Prediction — ML Pipeline
===============================================================
Dataset  : customer_acquisition_data.csv
Features : channel, cost, conversion_rate
Target   : revenue (proxy for CLV)
Models   : Linear Regression, Ridge, Random Forest, Gradient Boosting
===============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────
# 1. LOAD & ENGINEER FEATURES
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  CUSTOMER LIFETIME VALUE — ML PIPELINE")
print("=" * 60)

df = pd.read_csv(r'D:\DATASETS\customer_acquisition_data.csv')

print(f"\n[DATA]  Rows: {df.shape[0]}  |  Cols: {df.shape[1]}")
print(df.head())

# --- Feature Engineering ---
# CLV = Revenue - Acquisition Cost  (net value of the customer)
df['clv']             = df['revenue'] - df['cost']
# ROI on acquisition spend
df['roi']             = df['revenue'] / df['cost']
# Revenue per unit of conversion rate
df['revenue_per_cr']  = df['revenue'] / (df['conversion_rate'] + 1e-9)
# Cost efficiency
df['cost_efficiency'] = df['conversion_rate'] / df['cost']

print("\n[FEATURE ENGINEERING]  New columns added:")
print(df[['clv','roi','revenue_per_cr','cost_efficiency']].describe().round(2))

# --- Encode channel ---
le = LabelEncoder()
df['channel_enc'] = le.fit_transform(df['channel'])

print(f"\n[CHANNELS]  {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ─────────────────────────────────────────────────────────────
# 2. CLV SEGMENTATION  (Bronze / Silver / Gold / Platinum)
# ─────────────────────────────────────────────────────────────

percentiles = [0, 25, 50, 75, 100]
bins  = np.percentile(df['clv'], percentiles)
bins  = np.unique(bins)                         # drop duplicate edges if any
n_bins = len(bins) - 1
labels = ['Bronze', 'Silver', 'Gold', 'Platinum'][:n_bins]

df['clv_segment'] = pd.cut(df['clv'], bins=bins, labels=labels,
                            include_lowest=True)

print("\n[CLV SEGMENTS]")
print(df['clv_segment'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

FEATURES = ['cost', 'conversion_rate', 'channel_enc',
            'roi', 'cost_efficiency']
TARGET   = 'clv'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n[SPLIT]  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()

models = {
    "Linear Regression":   Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]),
    "Ridge Regression":    Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    "Lasso Regression":    Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=1.0))]),
    "Random Forest":       RandomForestRegressor(n_estimators=200, max_depth=8,
                                                  random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                      max_depth=4, random_state=42),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n[TRAINING MODELS]\n")
print(f"{'Model':<22}  {'CV-R²':>8}  {'Test-R²':>8}  {'MAE':>10}  {'RMSE':>10}")
print("-" * 65)

for name, model in models.items():
    # Cross-validation on training set
    cv_r2 = cross_val_score(model, X_train, y_train,
                             cv=kf, scoring='r2').mean()
    # Fit on full train, evaluate on test
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = {
        'model': model, 'preds': preds,
        'cv_r2': cv_r2, 'r2': r2, 'mae': mae, 'rmse': rmse
    }
    print(f"{name:<22}  {cv_r2:>8.4f}  {r2:>8.4f}  {mae:>10.2f}  {rmse:>10.2f}")

# Best model by test R²
best_name = max(results, key=lambda n: results[n]['r2'])
best      = results[best_name]
print(f"\n[BEST MODEL]  {best_name}  (Test R² = {best['r2']:.4f})")

# ─────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

best_model = best['model']
# Use permutation importance (works for all model types)
pi = permutation_importance(best_model, X_test, y_test,
                             n_repeats=20, random_state=42)
feat_imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)

print("\n[FEATURE IMPORTANCE — Permutation]")
print(feat_imp.round(4))

# ─────────────────────────────────────────────────────────────
# 6. CLV PREDICTIONS DATAFRAME
# ─────────────────────────────────────────────────────────────

test_df = X_test.copy()
test_df['actual_clv']    = y_test.values
test_df['predicted_clv'] = best['preds'].round(2)
test_df['error']         = (test_df['predicted_clv'] - test_df['actual_clv']).round(2)
test_df['channel']       = le.inverse_transform(test_df['channel_enc'].astype(int))

print("\n[PREDICTION SAMPLE — First 10 test rows]")
print(test_df[['channel','cost','conversion_rate',
               'actual_clv','predicted_clv','error']].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. CHANNEL CLV ANALYSIS
# ─────────────────────────────────────────────────────────────

channel_stats = df.groupby('channel').agg(
    count        = ('clv', 'count'),
    mean_clv     = ('clv', 'mean'),
    median_clv   = ('clv', 'median'),
    total_clv    = ('clv', 'sum'),
    mean_roi     = ('roi', 'mean'),
    mean_conv    = ('conversion_rate', 'mean'),
).round(2)

print("\n[CHANNEL CLV SUMMARY]")
print(channel_stats.to_string())

# ─────────────────────────────────────────────────────────────
# 8. VISUALISATIONS  (saved to PNG)
# ─────────────────────────────────────────────────────────────

sns.set_theme(style='whitegrid', palette='muted')
PALETTE = ['#2196F3','#FF5722','#4CAF50','#9C27B0']

fig = plt.figure(figsize=(20, 24))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 8a: CLV Distribution by Channel ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for i, ch in enumerate(df['channel'].unique()):
    subset = df[df['channel'] == ch]['clv']
    ax1.hist(subset, bins=20, alpha=0.65, label=ch, color=PALETTE[i])
ax1.set_title('CLV Distribution by Channel', fontsize=13, fontweight='bold')
ax1.set_xlabel('CLV ($)')
ax1.set_ylabel('Frequency')
ax1.legend(fontsize=8)

# ── 8b: Mean CLV & ROI by Channel ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
x   = np.arange(len(channel_stats))
w   = 0.35
bars1 = ax2.bar(x - w/2, channel_stats['mean_clv'],   width=w, label='Mean CLV',  color='#2196F3')
ax2b  = ax2.twinx()
bars2 = ax2b.bar(x + w/2, channel_stats['mean_roi'],   width=w, label='Mean ROI',  color='#FF5722', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(channel_stats.index, rotation=15, ha='right', fontsize=9)
ax2.set_ylabel('Mean CLV ($)', color='#2196F3')
ax2b.set_ylabel('Mean ROI', color='#FF5722')
ax2.set_title('Mean CLV & ROI per Channel', fontsize=13, fontweight='bold')
lines = [plt.Line2D([0],[0],color='#2196F3',lw=4),
         plt.Line2D([0],[0],color='#FF5722',lw=4)]
ax2.legend(lines, ['Mean CLV','Mean ROI'], loc='upper left', fontsize=8)

# ── 8c: Actual vs Predicted CLV ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(test_df['actual_clv'], test_df['predicted_clv'],
            alpha=0.5, color='#4CAF50', edgecolors='white', linewidth=0.4)
mn, mx = test_df['actual_clv'].min(), test_df['actual_clv'].max()
ax3.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
ax3.set_xlabel('Actual CLV ($)')
ax3.set_ylabel('Predicted CLV ($)')
ax3.set_title(f'Actual vs Predicted CLV\n{best_name}  |  R² = {best["r2"]:.4f}',
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=8)

# ── 8d: Model Comparison ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
model_names = list(results.keys())
r2_scores   = [results[n]['r2'] for n in model_names]
colors      = ['#FFB74D' if n != best_name else '#4CAF50' for n in model_names]
bars = ax4.barh(model_names, r2_scores, color=colors, edgecolor='white')
ax4.set_xlabel('Test R²')
ax4.set_title('Model Comparison (Test R²)', fontsize=13, fontweight='bold')
ax4.set_xlim(0, 1)
for bar, val in zip(bars, r2_scores):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)

# ── 8e: Feature Importance ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
feat_imp_sorted = feat_imp.sort_values()
feat_imp_sorted.plot(kind='barh', ax=ax5, color='#9C27B0', edgecolor='white')
ax5.set_title(f'Feature Importance\n(Permutation — {best_name})',
              fontsize=13, fontweight='bold')
ax5.set_xlabel('Mean Importance')

# ── 8f: CLV Segment Distribution ─────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
seg_counts = df['clv_segment'].value_counts().sort_index()
wedge_colors = ['#CD7F32','#C0C0C0','#FFD700','#E5E4E2']
ax6.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
        colors=wedge_colors[:len(seg_counts)], startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax6.set_title('CLV Segment Distribution', fontsize=13, fontweight='bold')

# ── 8g: Residuals ────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
residuals = test_df['actual_clv'] - test_df['predicted_clv']
ax7.scatter(test_df['predicted_clv'], residuals,
            alpha=0.5, color='#2196F3', edgecolors='white', linewidth=0.3)
ax7.axhline(0, color='red', linestyle='--', lw=1.5)
ax7.set_xlabel('Predicted CLV ($)')
ax7.set_ylabel('Residual ($)')
ax7.set_title('Residual Plot', fontsize=13, fontweight='bold')

# ── 8h: Correlation Heatmap ──────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
corr_cols = ['cost', 'conversion_rate', 'revenue', 'clv', 'roi', 'cost_efficiency']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax8, linewidths=0.5)
ax8.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
ax8.tick_params(labelsize=8)

fig.suptitle('Customer Lifetime Value — ML Analysis Dashboard',
             fontsize=16, fontweight='bold', y=1.01)

out_path = r'D:\DATASETS\clv_ml_dashboard.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[SAVED]  Dashboard → {out_path}")

# ─────────────────────────────────────────────────────────────
# 9. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY REPORT")
print("=" * 60)
print(f"\n  Dataset           : {df.shape[0]} customers, {df.shape[1]} columns")
print(f"  Target (CLV) mean : ${df['clv'].mean():.2f}")
print(f"  Target (CLV) std  : ${df['clv'].std():.2f}")
print(f"  CLV range         : ${df['clv'].min():.2f}  →  ${df['clv'].max():.2f}")

print(f"\n  Best Model        : {best_name}")
print(f"  CV R² (5-fold)    : {best['cv_r2']:.4f}")
print(f"  Test R²           : {best['r2']:.4f}")
print(f"  MAE               : ${best['mae']:.2f}")
print(f"  RMSE              : ${best['rmse']:.2f}")

print(f"\n  Top Feature       : {feat_imp.idxmax()}")
print(f"\n  Best Channel (CLV): {channel_stats['mean_clv'].idxmax()}")
print(f"  Best Channel (ROI): {channel_stats['mean_roi'].idxmax()}")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
