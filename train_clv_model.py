import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter, ParetoNBDFitter
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.plotting import plot_period_transactions, plot_frequency_recency_matrix
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading data...")
df = pd.read_csv('customer_lifetime_value_prediction.csv')
print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")

# Display basic information
print("\nDataset info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

# Visualize the distribution of key features
print("\nGenerating visualizations...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df['frequency'], bins=30, kde=True)
plt.title('Frequency Distribution')

plt.subplot(2, 2, 2)
sns.histplot(df['recency'], bins=30, kde=True)
plt.title('Recency Distribution')

plt.subplot(2, 2, 3)
sns.histplot(df['T'], bins=30, kde=True)
plt.title('Customer Age (T) Distribution')

plt.subplot(2, 2, 4)
sns.histplot(df['monetary_value'], bins=30, kde=True)
plt.title('Monetary Value Distribution')

plt.tight_layout()
plt.savefig('data_distributions.png')
print("Saved data distributions plot as 'data_distributions.png'")

# Train-test split
print("\nSplitting data into calibration and holdout sets...")
calibration_end_date = df['T'].max() * 0.7  # 70% for calibration
calibration_data = df[df['T'] <= calibration_end_date]
holdout_data = df[df['T'] > calibration_end_date]

print(f"Calibration period: 0 to {calibration_end_date:.0f} days")
print(f"Holdout period: {calibration_end_date:.0f} to {df['T'].max():.0f} days")
print(f"Calibration data shape: {calibration_data.shape}")
print(f"Holdout data shape: {holdout_data.shape}")

# Train the BG/NBD model
print("\nTraining BG/NBD model...")
bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(calibration_data['frequency'], 
        calibration_data['recency'], 
        calibration_data['T'],
        verbose=True)

# Train the Pareto/NBD model
print("\nTraining Pareto/NBD model...")
pnbd = ParetoNBDFitter(penalizer_coef=0.1)
pnbd.fit(calibration_data['frequency'], 
         calibration_data['recency'], 
         calibration_data['T'],
         verbose=True)

# Plot frequency/recency matrix for BG/NBD and Pareto/NBD
plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
plot_frequency_recency_matrix(bgf, max_frequency=20, max_recency=365)
plt.title('BG/NBD - Frequency-Recency Matrix')

plt.subplot(1, 2, 2)
plot_frequency_recency_matrix(pnbd, max_frequency=20, max_recency=365)
plt.title('Pareto/NBD - Frequency-Recency Matrix')

plt.tight_layout()
plt.savefig('frequency_recency_matrices.png')
plt.show()
print("Saved frequency-recency matrices as 'frequency_recency_matrices.png'")

# Train the Gamma-Gamma model
print("\nTraining Gamma-Gamma model...")
ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(calibration_data['frequency'], 
        calibration_data['monetary_value'],
        verbose=True)

# Make predictions on calibration data
print("\nMaking predictions on calibration data...")
prediction_days = 30
calibration_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    prediction_days,
    calibration_data['frequency'],
    calibration_data['recency'],
    calibration_data['T']
)

calibration_data['predicted_clv'] = ggf.customer_lifetime_value(
    bgf,
    calibration_data['frequency'],
    calibration_data['recency'],
    calibration_data['T'],
    calibration_data['monetary_value'],
    time=prediction_days/30,  # Convert to months
    freq='D'  # Daily frequency
)

# Make predictions on holdout data
print("Making predictions on holdout data...")
holdout_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    prediction_days,
    holdout_data['frequency'],
    holdout_data['recency'],
    holdout_data['T']
)

holdout_data['predicted_clv'] = ggf.customer_lifetime_value(
    bgf,
    holdout_data['frequency'],
    holdout_data['recency'],
    holdout_data['T'],
    holdout_data['monetary_value'],
    time=prediction_days/30,
    freq='D'
)

# Pareto/NBD predictions
holdout_data['pnbd_predicted'] = pnbd.conditional_expected_number_of_purchases_up_to_time(
    30,  # 30 days for comparison
    holdout_data['frequency'],
    holdout_data['recency'],
    holdout_data['T']
)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(holdout_data['actual30'], holdout_data['predicted_purchases']))
r2 = r2_score(holdout_data['actual30'], holdout_data['predicted_purchases'])

print(f"\nModel Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted for both models
plt.figure(figsize=(14, 6))

# BG/NBD predictions
plt.subplot(1, 2, 1)
plt.scatter(holdout_data['actual30'], holdout_data['predicted_purchases'], alpha=0.5)
plt.plot([0, holdout_data['actual30'].max()], [0, holdout_data['actual30'].max()], 'r--')
plt.xlabel('Actual Purchases')
plt.ylabel('Predicted Purchases')
plt.title('BG/NBD: Actual vs Predicted Purchases (30 days)')

# Pareto/NBD predictions
plt.subplot(1, 2, 2)
plt.scatter(holdout_data['actual30'], holdout_data['pnbd_predicted'], alpha=0.5, color='green')
plt.plot([0, holdout_data['actual30'].max()], [0, holdout_data['actual30'].max()], 'r--')
plt.xlabel('Actual Purchases')
plt.ylabel('Predicted Purchases')
plt.title('Pareto/NBD: Actual vs Predicted Purchases (30 days)')

plt.tight_layout()
plt.savefig('actual_vs_predicted_comparison.png')
plt.show()
print("Saved actual vs predicted comparison plot as 'actual_vs_predicted_comparison.png'")

# Save the models
print("\nSaving models...")
joblib.dump(bgf, 'bgf_model.pkl')
joblib.dump(ggf, 'ggf_model.pkl')
joblib.dump(pnbd, 'pnbd_model.pkl')
print("Models saved successfully as 'bgf_model.pkl', 'ggf_model.pkl', and 'pnbd_model.pkl'")

print("\nTraining and evaluation complete!")
