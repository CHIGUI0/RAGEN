# Reward Heatmap Visualization (Progression)

To better understand the variance and selection across groups during training progression, the carousel below visualizes the individual rollout rewards for **Steps 0, 10, 20... up to 90** across all 8 groups for the `Qwen2.5-3B-Instruct` linear top-p run.

- **The Y-axis** represents the Groups, sorted downwards from Highest Standard Deviation to Lowest Standard Deviation.
- **The X-axis** represents the individual sample instances within each group, sorted left to right (highest to lowest reward).
- **The Intensity (Red)** indicates the raw Reward value.
- **The Blue Line** separates the "Kept" groups (top) from the "Dropped" groups (bottom) based on the linear `0.9` top-p threshold evaluated using standard deviations.

*Note: Since GitHub does not natively support animated carousels, these are formatted as collapsible sections for easy sequential viewing.*

<details open>
  <summary><b>Step 0</b></summary>
  <img src="plots/reward_heatmap_step_0.png" alt="Reward Heatmap Step 0" width="800">
</details>

<details>
  <summary><b>Step 10</b></summary>
  <img src="plots/reward_heatmap_step_10.png" alt="Reward Heatmap Step 10" width="800">
</details>

<details>
  <summary><b>Step 20</b></summary>
  <img src="plots/reward_heatmap_step_20.png" alt="Reward Heatmap Step 20" width="800">
</details>

<details>
  <summary><b>Step 30</b></summary>
  <img src="plots/reward_heatmap_step_30.png" alt="Reward Heatmap Step 30" width="800">
</details>

<details>
  <summary><b>Step 40</b></summary>
  <img src="plots/reward_heatmap_step_40.png" alt="Reward Heatmap Step 40" width="800">
</details>

<details>
  <summary><b>Step 50</b></summary>
  <img src="plots/reward_heatmap_step_50.png" alt="Reward Heatmap Step 50" width="800">
</details>

<details>
  <summary><b>Step 60</b></summary>
  <img src="plots/reward_heatmap_step_60.png" alt="Reward Heatmap Step 60" width="800">
</details>

<details>
  <summary><b>Step 70</b></summary>
  <img src="plots/reward_heatmap_step_70.png" alt="Reward Heatmap Step 70" width="800">
</details>

<details>
  <summary><b>Step 80</b></summary>
  <img src="plots/reward_heatmap_step_80.png" alt="Reward Heatmap Step 80" width="800">
</details>

<details>
  <summary><b>Step 90</b></summary>
  <img src="plots/reward_heatmap_step_90.png" alt="Reward Heatmap Step 90" width="800">
</details>
