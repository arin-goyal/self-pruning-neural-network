import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hardcoding the results we already obtained so we don't have to retrain!
lambdas_str = ['0.0', '0.0001', '0.001']
accuracies = [53.14, 50.90, 44.00]
sparsities = [0.00, 95.27, 99.78]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Accuracy on left Y axis
color = 'tab:blue'
ax1.set_xlabel('Lambda Penalty')
ax1.set_ylabel('Test Accuracy (%)', color=color, fontweight='bold')
ax1.plot(lambdas_str, accuracies, marker='o', color=color, linewidth=2, markersize=8)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 100)

# Plot Sparsity on right Y axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Sparsity Level (%)', color=color, fontweight='bold')
ax2.plot(lambdas_str, sparsities, marker='x', color=color, linewidth=2, linestyle='--', markersize=8)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)

plt.title('Accuracy vs. Sparsity Trade-off Across Lambdas')
fig.tight_layout()
plt.savefig('tradeoff_plot.png')
print("Trade-off plot generated successfully!")
