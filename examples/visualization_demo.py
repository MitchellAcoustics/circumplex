"""Circumplex Visualization Demo.

This script demonstrates the visualization capabilities of the `circumplex` package,
showing how to create publication-ready plots of SSM analysis results.

The package provides three main plot types:
- Circle plots: Show amplitude/displacement on circumplex circle
- Curve plots: Show fitted curves overlaid on observed scores
- Contrast plots: Show parameter differences between groups
"""

# %%
# ## Setup

import numpy as np

from circumplex import load_dataset, plot_circle, ssm_analyze

# Set random seed for reproducible results
np.random.seed(12345)

# %%
# ## Example 1: Single Profile - Circle Plot

# Load data and run analysis
aw2009 = load_dataset("aw2009")
results_single = ssm_analyze(
    aw2009,
    scales=list(range(8)),
    boots=1000,
    seed=12345,
)

# Display summary
results_single.summary()

# Create basic circle plot using SSM object method
fig = results_single.plot_circle()

# %%
# ## Example 2: Single Profile - Curve Plot

# Create curve plot showing fitted curve overlaid on observed scores
fig = results_single.plot_curve()

# %%
# ## Example 3: Customized Single Profile

# Customize circle plot appearance
fig = results_single.plot_circle(
    colors=None,  # Single blue color
    figsize=(10, 10),
    fontsize=14,
    title="Interpersonal Profile (aw2009 dataset)",
)


# %%
# ## Example 4: Multi-Group Comparison - Circle Plot

# Analyze by gender
jz2017 = load_dataset("jz2017")
results_groups = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    grouping="Gender",
    boots=1000,
    seed=12345,
)

# Plot comparison with default palette
fig = results_groups.plot_circle()

# %%
# ## Example 5: Multi-Group - Custom Colors

# Use custom colors for groups
fig = results_groups.plot_circle(
    colors=["#FF6B6B", "#4ECDC4"],  # Custom hex colors
    figsize=(10, 10),
    title="Gender Comparison: Interpersonal Profiles",
)

# %%
# ## Example 6: Multi-Group - Curve Plots

# Compare groups with curve plots
scale_names = ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"]
fig = results_groups.plot_curve(angle_labels=scale_names)

# %%
# ## Example 7: Group Contrast Analysis

# Include contrast in analysis
results_contrast = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    grouping="Gender",
    contrast=True,
    boots=1000,
    seed=12345,
)

# Plot all three profiles (Female, Male, Male - Female)
fig = results_contrast.plot_circle()

# Plot contrast differences
fig = results_contrast.plot_contrast()

# %%
# ## Example 8: Simplified Contrast Plot (Drop X/Y)

# Plot contrasts without X and Y parameters for simpler view
fig = results_contrast.plot_contrast(drop_xy=True)

# %%
# ## Example 9: Correlation-Based Profile

# Analyze correlation with personality disorder measure
results_corr = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures="PARPD",
    boots=1000,
    seed=12345,
)

# Plot correlation profile
fig = results_corr.plot_circle(
    colors=["purple"],
    title="PARPD Correlations with Interpersonal Scales",
)

# %%
# ## Example 10: Multi-Measure Comparison

# Compare two personality disorder measures
results_measures = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures=["ASPD", "NARPD"],
    boots=1000,
    seed=12345,
)

# Circle plot comparison
fig = results_measures.plot_circle(
    colors=["#E74C3C", "#3498DB"],
    title="Personality Disorder Profiles: ASPD vs NARPD",
)

# Curve plots comparison
fig = results_measures.plot_curve(
    angle_labels=scale_names,
    colors=["#E74C3C", "#3498DB"],
)

# %%
# ## Example 11: Measure Contrast

# Compare measures with contrast
results_measure_contrast = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures=["ASPD", "NARPD"],
    contrast=True,
    boots=1000,
    seed=12345,
)

# Plot measure contrast
fig = results_measure_contrast.plot_contrast(drop_xy=True)
# fig.savefig("examples/output/contrast_measures.png", dpi=300, bbox_inches="tight")  # noqa: ERA001

# %%
# ## Example 12: Profile Selection

# Plot only specific profiles from multi-profile results
# Using the imported plot_circle function directly

# Plot only first profile from measure results
fig = plot_circle(
    results_measures.results,
    results_measures.details.angles,
    profile_indices=[0],
    colors=["#E74C3C"],
    title="ASPD Profile Only",
)

# %%
# ## Example 13: Seaborn Palettes

# Try different seaborn color palettes
results_multi = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures=["ASPD", "NARPD", "BORPD"],
    boots=500,  # Fewer boots for speed
    seed=12345,
)

# Using 'husl' palette
fig = results_multi.plot_circle(colors="husl", title="HUSL Palette")

# Using 'colorblind' palette
fig = results_multi.plot_circle(colors="colorblind", title="Colorblind Palette")

# %%
# ## Visualization Features Summary

print(
    """
### Circumplex Visualization Features

✓ **Three Plot Types**:
  - Circle plots: Amplitude/displacement on circumplex circle
  - Curve plots: Fitted cosine curves overlaid on observed scores
  - Contrast plots: Parameter differences with confidence intervals

✓ **SSM Object Methods**: Convenient plotting directly from results
  - results.plot_circle()
  - results.plot_curve()
  - results.plot_contrast()

✓ **Customization**: Colors, sizes, fonts, titles
✓ **Confidence intervals**: Bootstrap CIs displayed as arc bars (circle)
  or error bars (contrast)
✓ **Publication-ready**: High-resolution output (300 dpi)
✓ **Flexible colors**: Seaborn palettes or custom color lists

### Plot Types in Detail

**Circle Plot (plot_circle)**:
- Shows amplitude and displacement on circular plot
- Profile points at (x, y) Cartesian coordinates
- Arc bars show confidence intervals for amplitude and displacement
- Supports single or multiple profiles
- Automatic legend for multiple profiles

**Curve Plot (plot_curve)**:
- Faceted plots (one subplot per profile)
- Fitted cosine curve overlaid on observed scale scores
- Black points and lines show observed data
- Colored curve shows SSM model fit
- Dashed curves indicate low fit (R² < 0.70)

**Contrast Plot (plot_contrast)**:
- Shows differences between two profiles/groups
- One subplot per SSM parameter (elevation, amplitude, displacement, etc.)
- Points colored by significance (CI excludes zero)
- Optional drop_xy parameter for simplified plot

### Customization Options

**Common Parameters**:
- colors: Seaborn palette name ('Set2', 'husl', etc.) or list of custom colors
- figsize: Figure dimensions in inches (width, height)
- fontsize/base_size: Font size for labels and text
- drop_lowfit: Omit profiles with fit < 0.70

**Circle Plot Specific**:
- amax: Maximum amplitude for scaling
- angle_labels: Custom labels for angles
- title: Plot title

**Curve Plot Specific**:
- angle_labels: Labels for x-axis (scale names)

**Contrast Plot Specific**:
- drop_xy: Omit x-value and y-value parameters
- sig_color: Color for significant contrasts
- ns_color: Color for non-significant contrasts

### Usage Patterns

**Method 1: SSM Object Methods (Recommended)**
```python
results = ssm_analyze(data, scales=[...])
fig = results.plot_circle()
fig = results.plot_curve()
```

**Method 2: Direct Function Calls (Advanced)**
```python
from circumplex import plot_circle, plot_curve, plot_contrast
fig = plot_circle(results.results, results.details.angles)
fig = plot_curve(results.results, results.scores, results.details.angles)
```

Both patterns support the same customization options.
"""
)
