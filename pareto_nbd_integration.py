

# 1. Import required libraries (if not already imported)
from lifetimes import ParetoNBDFitter
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_frequency_recency_matrix
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 2. Train the Pareto/NBD model
print("\nTraining Pareto/NBD model...")
pnbd = ParetoNBDFitter(penalizer_coef=0.1)
pnbd.fit(calibration_data['frequency'], 
         calibration_data['recency'], 
         calibration_data['T'],
         verbose=True)

# 3. Plot frequency/recency matrix for both models side by side
plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plot_frequency_recency_matrix(bgf, max_frequency=20, max_recency=365)
plt.title('BG/NBD - Frequency-Recency Matrix')

plt.subplot(1, 2, 2)
plot_frequency_recency_matrix(pnbd, max_frequency=20, max_recency=365)
plt.title('Pareto/NBD - Frequency-Recency Matrix')

plt.tight_layout()
plt.show()

# 4. Make predictions with Pareto/NBD on holdout data
print("Making predictions with Pareto/NBD on holdout data...")
holdout_data['pnbd_predicted'] = pnbd.conditional_expected_number_of_purchases_up_to_time(
    prediction_days,
    holdout_data['frequency'],
    holdout_data['recency'],
    holdout_data['T']
)

# 5. Calculate evaluation metrics for both models
print("\nModel Evaluation (30-day predictions):")

# BG/NBD metrics
bgf_rmse = np.sqrt(mean_squared_error(holdout_data['actual30'], holdout_data['predicted_purchases']))
bgf_r2 = r2_score(holdout_data['actual30'], holdout_data['predicted_purchases'])

# Pareto/NBD metrics
pnbd_rmse = np.sqrt(mean_squared_error(holdout_data['actual30'], holdout_data['pnbd_predicted']))
pnbd_r2 = r2_score(holdout_data['actual30'], holdout_data['pnbd_predicted'])

# Print metrics
eval_metrics = pd.DataFrame({
    'Model': ['BG/NBD', 'Pareto/NBD'],
    'RMSE': [bgf_rmse, pnbd_rmse],
    'R²': [bgf_r2, pnbd_r2]
})

print("\nModel Comparison:")
print(eval_metrics.round(4))

# 6. Plot actual vs predicted for both models
plt.figure(figsize=(14, 6))

# BG/NBD predictions
plt.subplot(1, 2, 1)
plt.scatter(holdout_data['actual30'], holdout_data['predicted_purchases'], alpha=0.5)
plt.plot([0, holdout_data['actual30'].max()], [0, holdout_data['actual30'].max()], 'r--')
plt.xlabel('Actual Purchases')
plt.ylabel('Predicted Purchases')
plt.title(f'BG/NBD\nRMSE: {bgf_rmse:.4f}, R²: {bgf_r2:.4f}')

# Pareto/NBD predictions
plt.subplot(1, 2, 2)
plt.scatter(holdout_data['actual30'], holdout_data['pnbd_predicted'], alpha=0.5, color='green')
plt.plot([0, holdout_data['actual30'].max()], [0, holdout_data['actual30'].max()], 'r--')
plt.xlabel('Actual Purchases')
plt.ylabel('Predicted Purchases')
plt.title(f'Pareto/NBD\nRMSE: {pnbd_rmse:.4f}, R²: {pnbd_r2:.4f}')

plt.tight_layout()
plt.show()

# 7. Save the Pareto/NBD model
print("\nSaving Pareto/NBD model...")
joblib.dump(pnbd, 'pnbd_model.pkl')
print("Pareto/NBD model saved successfully as 'pnbd_model.pkl'")

# 8. Add Pareto/NBD predictions to the calibration data for future use
calibration_data['pnbd_predicted'] = pnbd.conditional_expected_number_of_purchases_up_to_time(
    prediction_days,
    calibration_data['frequency'],
    calibration_data['recency'],
    calibration_data['T']
)

print("\nPareto/NBD model integration complete!")
