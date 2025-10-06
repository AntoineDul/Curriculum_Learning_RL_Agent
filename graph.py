import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
data = {
    "Lambda": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    "Plasticity": [0.005932807, 0.076030498, 0.146403282, 0.234834788,
                   0.325361818, 0.419632342, 0.504860042, 0.573592172,
                   0.611619639, 0.616087068]
}

df = pd.DataFrame(data)

# Set seaborn style
sns.set(style="whitegrid", context="talk")

# Create figure
plt.figure(figsize=(10, 6))

# Line plot with markers
sns.lineplot(x="Lambda", y="Plasticity", data=df, marker="o", linewidth=2, markersize=8, color="#1f77b4")

# Add title and labels
plt.title("Plasticity vs Lambda", fontsize=18, weight='bold')
plt.xlabel("Lambda", fontsize=14)
plt.ylabel("Plasticity", fontsize=14)

# Reverse x-axis if you want Lambda decreasing left to right
plt.gca().invert_xaxis()

# Optional: annotate each point with its value
for x, y in zip(df["Lambda"], df["Plasticity"]):
    plt.text(x, y + 0.02, f"{y:.3f}", ha='center', fontsize=10)

# Show grid
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Tight layout
plt.tight_layout()

# Show plot
plt.show()