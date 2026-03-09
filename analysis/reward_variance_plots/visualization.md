# Reward Heatmap Visualization (Progression)

To better understand the variance and selection across groups during training progression, the carousel below visualizes the individual rollout rewards for **Steps 0, 10, 20... up to 90** across all 8 groups for the `Qwen2.5-3B-Instruct` linear top-p run.

- **The Y-axis** represents the Groups, sorted downwards from Highest Standard Deviation to Lowest Standard Deviation.
- **The X-axis** represents the individual sample instances within each group, sorted left to right (highest to lowest reward).
- **The Intensity (Red)** indicates the raw Reward value.
- **The Blue Line** separates the "Kept" groups (top) from the "Dropped" groups (bottom) based on the linear `0.9` top-p threshold evaluated using standard deviations.

*Note: If viewing on GitHub, scroll down to see the images sequentially. If viewing in an editor that supports the `carousel` markdown extension, you can click through them interactively.*

````carousel
![Reward Heatmap Step 0](plots/reward_heatmap_step_0.png)
<!-- slide -->
![Reward Heatmap Step 10](plots/reward_heatmap_step_10.png)
<!-- slide -->
![Reward Heatmap Step 20](plots/reward_heatmap_step_20.png)
<!-- slide -->
![Reward Heatmap Step 30](plots/reward_heatmap_step_30.png)
<!-- slide -->
![Reward Heatmap Step 40](plots/reward_heatmap_step_40.png)
<!-- slide -->
![Reward Heatmap Step 50](plots/reward_heatmap_step_50.png)
<!-- slide -->
![Reward Heatmap Step 60](plots/reward_heatmap_step_60.png)
<!-- slide -->
![Reward Heatmap Step 70](plots/reward_heatmap_step_70.png)
<!-- slide -->
![Reward Heatmap Step 80](plots/reward_heatmap_step_80.png)
<!-- slide -->
![Reward Heatmap Step 90](plots/reward_heatmap_step_90.png)
````
