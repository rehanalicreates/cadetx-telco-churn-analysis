# Task 5: Decision Threshold Modeling (no seaborn)
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

# df and rf already exist from previous tasks
feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[feature_cols]
y = df['Churn_Numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_probs = rf.predict_proba(X_test)[:, 1]

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"Precision: {precisions[optimal_idx]:.3f}")
print(f"Recall: {recalls[optimal_idx]:.3f}")
print(f"F1 Score: {f1_scores[optimal_idx]:.3f}")

# Business cost analysis
cost_retain = 50
value_retained = 500

print("\n--- Cost-Benefit Analysis ---")
print(f"{'Threshold':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Net Benefit':<12}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_probs >= t).astype(int)
    tp = np.sum((y_pred_t == 1) & (y_test.values == 1))
    fp = np.sum((y_pred_t == 1) & (y_test.values == 0))
    fn = np.sum((y_pred_t == 0) & (y_test.values == 1))
    cost = (tp + fp) * cost_retain
    benefit = tp * value_retained
    net = benefit - cost
    p = tp/(tp+fp+1e-10)
    r = tp/(tp+fn+1e-10)
    f1 = 2*p*r/(p+r+1e-10)
    print(f"{t:<12.1f} {p:<10.3f} {r:<10.3f} {f1:<10.3f} Rs.{net:<10,}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()
