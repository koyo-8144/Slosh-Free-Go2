def percent_improvement(baseline_stats, new_stats):
    """
    Calculates the percent improvement for mean, std, and max error values.
    Lower values are considered better.
    """
    improvement = {}
    for key in ("mean", "std", "max"):
        base = baseline_stats[key]
        new = new_stats[key]
        improvement[key] = 100.0 * (base - new) / abs(base)
    return improvement

# Tilt Error
# baseline_stats = {
#     "mean": 3.314567,
#     "std": 2.647482,
#     "max": 14.467972,
# }
# Linvel X Error
baseline_stats = {
    "mean": 0.132443,
    "std": 0.191960,
    "max": 1.213121,
}
# # Linvel Y Error
# baseline_stats = {
#     "mean": 0.034213,
#     "std": 0.027658,
#     "max": 0.193207,
# }
# # Angvel Z Error
# baseline_stats = {
#     "mean": 0.058430,
#     "std": 0.045432,
#     "max": 0.322728,
# }


new_stats = {
    "mean": 0.060342,
    "std": 0.051075,
    "max": 0.755205,
}

# Compute improvements
improvement = percent_improvement(baseline_stats, new_stats)

# Display results
print(f"Mean Error Improvement: {improvement['mean']:+.2f}%")
print(f"Std  Error Improvement: {improvement['std']:+.2f}%")
print(f"Max  Error Improvement: {improvement['max']:+.2f}%")
