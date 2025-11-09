"""
# Structural Summary Method (SSM) Analysis Demo.

This notebook demonstrates the core functionality of the `circumplex` package
for analyzing circumplex data using the Structural Summary Method (SSM).

The SSM provides six key parameters that summarize circular data patterns:
- **Elevation ($e$)**: Mean level across all scales
- **X-value ($x$)**: Projection onto the x-axis (cosine component)
- **Y-value ($y$)**: Projection onto the y-axis (sine component)
- **Amplitude ($a$)**: Vector length (prototypicality of the pattern)
- **Displacement ($d$)**: Angular position in degrees $[0, 360)$
- **Fit ($R^2$)**: Proportion of variance explained by the model

Bootstrap confidence intervals are computed for all parameters except fit.
"""

# %%
# ## Setup

import numpy as np
from rich.console import Console

from circumplex import load_dataset, ssm_analyze

console = Console()

# Set random seed for reproducible bootstrap results
np.random.seed(12345)

# %%
# ## Example 1: Single-Group Mean-Based Analysis

# Load a small example dataset (5 observations, 8 octant scales)
aw2009 = load_dataset("aw2009")
console.print(f"[bold]aw2009 dataset shape:[/bold] {aw2009.shape}")
console.print("\n[bold]First few rows:[/bold]")
console.print(aw2009.head())

# %%
# ### Run SSM Analysis
#
# For mean-based analysis, we analyze the profile of mean scale scores.
# The `ssm_analyze()` function automatically uses octant angles [90, 135, 180, 225, 270, 315, 360, 45]
# when 8 scales are provided and angles=None.

results_single = ssm_analyze(
    aw2009,
    scales=list(range(8)),  # Use all 8 scales (or specify column names)
    boots=2000,  # Number of bootstrap resamples
    interval=0.95,  # 95% confidence intervals
    seed=12345,  # For reproducibility
)

console.print("\n[bold]SSM Parameters:[/bold]")
console.print(
    results_single.results[
        ["Label", "e_est", "x_est", "y_est", "a_est", "d_est", "fit_est"]
    ]
)

console.print("\n[bold]Confidence Intervals:[/bold]")
console.print(
    results_single.results[
        ["Label", "e_lci", "e_uci", "a_lci", "a_uci", "d_lci", "d_uci"]
    ]
)

# %%
# ### Interpretation
#
# - **Elevation (e_est)**: 0.423 - The mean level across all scales is positive
# - **Amplitude (a_est)**: 0.981 - Strong prototypicality (high consistency with circular model)
# - **Displacement (d_est)**: 344.4° - The peak is near 360° (quadrant IV)
# - **Fit (fit_est)**: 0.954 - The circular model explains 95.4% of variance
#
# The 95% CIs show the uncertainty in these estimates based on bootstrap resampling.

# %%
# ## Example 2: Multi-Group Mean-Based Analysis

# Load a larger dataset with multiple groups
jz2017 = load_dataset("jz2017")
console.print(f"[bold]jz2017 dataset shape:[/bold] {jz2017.shape}")
console.print(f"\n[bold]Columns:[/bold] {jz2017.columns.tolist()}")
console.print("\n[bold]Gender distribution:[/bold]")
console.print(jz2017["Gender"].value_counts())

# %%
# ### Compare Groups
#
# Analyze interpersonal circumplex profiles separately for females and males.

results_groups = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),  # Columns 1-8 (PA, BC, DE, FG, HI, JK, LM, NO)
    grouping="Gender",  # Split by gender
    boots=2000,
    seed=12345,
)

console.print("\n[bold]SSM Parameters by Group:[/bold]")
console.print(
    results_groups.results[
        ["Label", "e_est", "x_est", "y_est", "a_est", "d_est", "fit_est"]
    ]
)

console.print("\n[bold]Mean Scale Scores by Group:[/bold]")
console.print(results_groups.scores)

# %%
# ### Group Comparison
#
# Both groups show similar displacement (around 320-326°) but differ in amplitude:
# - **Female**: Amplitude = 0.553 (moderate prototypicality)
# - **Male**: Amplitude = 0.299 (lower prototypicality)
#
# This suggests females show a more consistent interpersonal pattern than males
# in this sample.

# %%
# ## Example 3: Multi-Group Analysis with Contrast

# Compute the difference between groups (Male - Female)
results_contrast = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    grouping="Gender",
    contrast=True,  # Add contrast row
    boots=2000,
    seed=12345,
)

console.print("\n[bold]SSM Parameters with Contrast:[/bold]")
console.print(
    results_contrast.results[["Label", "e_est", "x_est", "y_est", "a_est", "d_est"]]
)

console.print("\n[bold]Contrast Interpretation:[/bold]")
contrast_row = results_contrast.results.iloc[2]
console.print(f"  Elevation difference: {contrast_row['e_est']:.3f}")
console.print(f"  Amplitude difference: {contrast_row['a_est']:.3f}")
console.print(f"  Angular difference: {contrast_row['d_est']:.3f}°")

# %%
# ### Contrast Interpretation
#
# The contrast row shows Male - Female differences:
# - **Elevation**: -0.062 (males slightly lower overall)
# - **Amplitude**: -0.254 (males show much less prototypicality)
# - **Displacement**: -5.28° (minimal angular difference)
#
# The key finding is that males have significantly lower amplitude, indicating
# their interpersonal patterns are less differentiated/prototypical.

# %%
# ## Example 4: Correlation-Based Analysis

# Analyze how personality disorder symptoms (PARPD) relate to interpersonal scales
results_corr = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures="PARPD",  # Paranoid personality disorder scale
    boots=2000,
    seed=12345,
)

console.print("\n[bold]Correlation-Based SSM Parameters:[/bold]")
console.print(
    results_corr.results[
        ["Label", "e_est", "x_est", "y_est", "a_est", "d_est", "fit_est"]
    ]
)

console.print("\n[bold]Correlation Profile:[/bold]")
console.print(results_corr.scores)

# %%
# ### Correlation Interpretation
#
# The correlation profile shows how PARPD symptoms relate to interpersonal behavior:
# - **Elevation (e_est)**: 0.250 - PARPD is positively correlated with interpersonal problems overall
# - **Amplitude (a_est)**: 0.150 - Moderate differentiation in the correlation pattern
# - **Displacement (d_est)**: 128.9° - Peak correlation around 135° (Quadrant II)
# - **Fit (fit_est)**: 0.802 - The circular model explains 80% of the correlation variance
#
# This suggests PARPD is most strongly associated with interpersonal behaviors
# in the cold-dominant quadrant (quadrant II).

# %%
# ## Example 5: Multi-Measure Correlation with Contrast

# Compare correlation profiles for two different personality disorders
results_measure_contrast = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures=["ASPD", "NARPD"],  # Antisocial and Narcissistic PD
    contrast=True,
    boots=2000,
    seed=12345,
)

console.print("\n[bold]Multi-Measure SSM Parameters:[/bold]")
console.print(
    results_measure_contrast.results[["Label", "e_est", "a_est", "d_est", "fit_est"]]
)

console.print("\n[bold]Measure Contrast:[/bold]")
contrast_row = results_measure_contrast.results.iloc[2]
console.print(f"  Elevation difference (NARPD - ASPD): {contrast_row['e_est']:.3f}")
console.print(f"  Amplitude difference: {contrast_row['a_est']:.3f}")
console.print(f"  Angular difference: {contrast_row['d_est']:.1f}°")

# %%
# ### Multi-Measure Interpretation
#
# Comparing ASPD vs NARPD interpersonal profiles:
# - **ASPD**: Displacement at ~315° (dominant-cold), moderate amplitude
# - **NARPD**: Displacement at ~82° (dominant-warm), higher amplitude
# - **Contrast**: NARPD shows 0.148 higher amplitude (more differentiated pattern)
#   and is ~127° apart (nearly opposite on the circle)
#
# This reveals distinct interpersonal correlates: ASPD associates with cold-dominance
# while NARPD associates with warm-dominance.

# %%
# ## Example 6: Group x Measure Correlation

# Analyze how NARPD relates to interpersonal behavior differently by gender
results_group_corr = ssm_analyze(
    jz2017,
    scales=list(range(1, 9)),
    measures="NARPD",
    grouping="Gender",
    contrast=True,  # Compare genders
    boots=2000,
    seed=12345,
)

console.print("\n[bold]Group x Measure SSM Parameters:[/bold]")
console.print(
    results_group_corr.results[["Label", "e_est", "a_est", "d_est", "fit_est"]]
)

# %%
# ### Group x Measure Interpretation
#
# NARPD-interpersonal relationships by gender:
# - **Female**: Higher elevation (0.334) and amplitude (0.231)
# - **Male**: Lower elevation (0.281) and amplitude (0.160)
# - **Contrast**: Females show stronger and more differentiated NARPD-interpersonal associations
#
# Both groups have similar angular positions (~73-82°), suggesting the _location_
# of NARPD correlates is similar, but the _strength_ differs by gender.

# %%
# ## Example 7: Custom Angles

# You can specify custom angular positions for non-octant circumplex models
# For example, using 4 scales at quadrant positions:

results_custom = ssm_analyze(
    jz2017,
    scales=["PA", "DE", "HI", "LM"],  # 4 scales at quadrant positions
    angles=[90, 180, 270, 360],  # Quadrant angles in degrees
    boots=1000,  # Fewer boots for speed
    seed=12345,
)

console.print("\n[bold]Custom Angles SSM Parameters:[/bold]")
console.print(results_custom.results[["Label", "e_est", "a_est", "d_est", "fit_est"]])

# %%
# ## Key Features Summary

console.print("""
[bold cyan]circumplex Package Features[/bold cyan]

✓ [green]Mean-based SSM:[/green] Analyze profiles of mean scale scores
✓ [green]Correlation-based SSM:[/green] Analyze how measures correlate with circumplex scales
✓ [green]Multi-group designs:[/green] Compare profiles across groups (e.g., gender, diagnosis)
✓ [green]Contrast analysis:[/green] Test differences between 2 groups or 2 measures
✓ [green]Bootstrap CIs:[/green] Percentile confidence intervals via resampling
✓ [green]Circular statistics:[/green] Proper handling of angular data
✓ [green]Flexible angles:[/green] Support for any circumplex configuration (OCTANTS, QUADRANTS, etc.)
✓ [green]Numerical parity:[/green] Results match R circumplex package to 3+ decimal places

[bold cyan]Analysis Types Supported[/bold cyan]

1. Single-group mean profile
2. Multi-group mean profiles
3. Multi-group mean profiles with contrast
4. Single-group correlation profile
5. Multi-group correlation profiles
6. Multi-measure correlation profiles with contrast
7. Group x measure correlation with contrast

[bold cyan]Bootstrap Implementation[/bold cyan]

• Stratified sampling when groups are present
• Circular quantile method for displacement CIs
• Listwise or pairwise deletion options
• Reproducible with seed parameter

[bold cyan]Next Steps[/bold cyan]

• Visualization components (coming soon)
• Instrument registry for standard scales (coming soon)
• Export/reporting functions (coming soon)
""")

# %%
# ## Technical Notes

console.print("""
[bold cyan]Parameter Calculation Details[/bold cyan]

The SSM parameters are calculated using the following formulas:

[bold]Elevation:[/bold] e = mean(scores)

[bold]X-value:[/bold] x = (2/n) x Σ(scores x cos(angles))

[bold]Y-value:[/bold] y = (2/n) x Σ(scores x sin(angles))

[bold]Amplitude:[/bold] a = √(x² + y²)

[bold]Displacement:[/bold] d = arctan2(y, x) [converted to degrees, [0, 360)]

[bold]Fit:[/bold] R² = 1 - (SS_residual / SS_total)

where predicted scores = e + a x cos(angles - d)

[bold cyan]Bootstrap Confidence Intervals[/bold cyan]

CIs are computed using the percentile method:
• Lower CI: 2.5th percentile of bootstrap distribution (for 95% CI)
• Upper CI: 97.5th percentile of bootstrap distribution

For displacement (circular data), a special circular quantile method is used
that accounts for angular wrapping at 0°/360°.

[bold cyan]Contrast Calculations[/bold cyan]

Contrasts are computed as second minus first (e.g., Male - Female):
• For linear parameters: simple subtraction
• For displacement: circular distance using angle_dist()
  - Returns shortest angular distance in [-180°, 180°]

[bold cyan]Data Requirements[/bold cyan]

• [bold]Scales:[/bold] Must be equally spaced around the circle (or specify custom angles)
• [bold]Sample size:[/bold] Minimum ~30 for stable bootstrap CIs (n=5 works but CIs are wide)
• [bold]Missing data:[/bold] Listwise deletion (default) or pairwise deletion available
• [bold]Grouping:[/bold] Groups are treated as categorical factors (alphabetical ordering)
""")

# %%
console.print(
    "\n✓ [bold green]Demo complete! The circumplex package is ready for SSM analysis.[/bold green]"
)
