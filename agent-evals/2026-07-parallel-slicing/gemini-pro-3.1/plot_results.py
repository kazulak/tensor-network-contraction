import matplotlib.pyplot as plt

workers = [1, 2, 4, 8]
time = [17.8344, 10.8477, 9.4121, 10.2356]
speedup = [1.02, 1.67, 1.92, 1.77]
efficiency = [1.02, 0.84, 0.48, 0.22]

fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:red'
ax1.set_xlabel('Number of Workers')
ax1.set_ylabel('Speedup', color=color)
ax1.plot(workers, speedup, marker='o', color=color, label='Speedup')
ax1.plot(workers, workers, linestyle='--', color='gray', label='Ideal Speedup')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Parallel Efficiency', color=color)  
ax2.plot(workers, efficiency, marker='s', linestyle='-.', color=color, label='Efficiency')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Parallel Tensor Network Contraction Performance (12x12 Grid, D=3)')
fig.tight_layout()  
plt.grid(True)
plt.savefig('speedup_plot.png')
plt.show()
