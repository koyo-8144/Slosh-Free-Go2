import re
import math
from pathlib import Path


TILT_STATS = 0
LINVEL_X_STATS = 0
LINVEL_Y_STATS = 0
ANGVEL_Z_STATS = 1


# 1) Point this to your folder containing the run*_tilt_stats.txt files:
# DATA_DIR = Path.home() / "Genesis" / "results" / "20250422_180004" / "tilt_stats"
DATA_DIR = Path("")

# 2) Prepare containers
mus     = []   # μ_i
sigmas  = []   # σ_i
maxima  = []   # M_i
ns      = []   # n_i

# 3) Regex patterns to extract the numbers
mean_pat   = re.compile(r"Mean.*:\s*([0-9]+\.[0-9]+)")
std_pat    = re.compile(r"Standard Deviation.*:\s*([0-9]+\.[0-9]+)")
max_pat    = re.compile(r"Maximum.*:\s*([0-9]+\.[0-9]+)")
n_pat      = re.compile(r"Total Step Length.*:\s*(\d+)")

# 4) Iterate over all your run files
if TILT_STATS:
    for fn in sorted(DATA_DIR.glob("*run*_tilt_stats.txt")):
        text = fn.read_text()
        try:
            mu_i     = float(mean_pat.search(text).group(1))
            sigma_i  = float(std_pat.search(text).group(1))
            M_i      = float(max_pat.search(text).group(1))
            n_i      = int(  n_pat.search(text).group(1))
        except AttributeError:
            print(f"Failed to parse {fn.name}, skipping.")
            continue

        print("mu_i: ", mu_i)
        print("sigma_i: ", sigma_i)
        print("M_i: ", M_i)
        print("n_i: ", n_i)

        mus.append(mu_i)
        sigmas.append(sigma_i)
        maxima.append(M_i)
        ns.append(n_i)
elif LINVEL_X_STATS:
    for fn in sorted(DATA_DIR.glob("*run*_linvel_x_stats.txt")):
        text = fn.read_text()
        try:
            mu_i     = float(mean_pat.search(text).group(1))
            sigma_i  = float(std_pat.search(text).group(1))
            M_i      = float(max_pat.search(text).group(1))
            n_i      = int(  n_pat.search(text).group(1))
        except AttributeError:
            print(f"Failed to parse {fn.name}, skipping.")
            continue

        print("mu_i: ", mu_i)
        print("sigma_i: ", sigma_i)
        print("M_i: ", M_i)
        print("n_i: ", n_i)

        mus.append(mu_i)
        sigmas.append(sigma_i)
        maxima.append(M_i)
        ns.append(n_i)
elif LINVEL_Y_STATS:
    for fn in sorted(DATA_DIR.glob("*run*_linvel_y_stats.txt")):
        text = fn.read_text()
        try:
            mu_i     = float(mean_pat.search(text).group(1))
            sigma_i  = float(std_pat.search(text).group(1))
            M_i      = float(max_pat.search(text).group(1))
            n_i      = int(  n_pat.search(text).group(1))
        except AttributeError:
            print(f"Failed to parse {fn.name}, skipping.")
            continue

        print("mu_i: ", mu_i)
        print("sigma_i: ", sigma_i)
        print("M_i: ", M_i)
        print("n_i: ", n_i)

        mus.append(mu_i)
        sigmas.append(sigma_i)
        maxima.append(M_i)
        ns.append(n_i)
elif ANGVEL_Z_STATS:
    for fn in sorted(DATA_DIR.glob("*run*_angvel_z_stats.txt")):
        text = fn.read_text()
        try:
            mu_i     = float(mean_pat.search(text).group(1))
            sigma_i  = float(std_pat.search(text).group(1))
            M_i      = float(max_pat.search(text).group(1))
            n_i      = int(  n_pat.search(text).group(1))
        except AttributeError:
            print(f"Failed to parse {fn.name}, skipping.")
            continue

        print("mu_i: ", mu_i)
        print("sigma_i: ", sigma_i)
        print("M_i: ", M_i)
        print("n_i: ", n_i)

        mus.append(mu_i)
        sigmas.append(sigma_i)
        maxima.append(M_i)
        ns.append(n_i)

# 5) Compute the pooled mean:
total_n = sum(ns)
pooled_mean = sum(n_i * mu_i for n_i, mu_i in zip(ns, mus)) / total_n

# 6) Compute the pooled variance:
#    sum over i of [ (n_i-1)*sigma_i^2  +  n_i*(mu_i - pooled_mean)^2 ]
numer = sum((n_i - 1) * sigma_i**2 + n_i * (mu_i - pooled_mean)**2
            for n_i, sigma_i, mu_i in zip(ns, sigmas, mus))
denom = total_n - 1
pooled_var = numer / denom
pooled_std = math.sqrt(pooled_var)

# 7) Compute the overall maximum
overall_max = max(maxima)

if TILT_STATS:
    print("Tilt")
elif LINVEL_X_STATS:
    print("Linear Velocity X")
elif LINVEL_Y_STATS:
    print("Linear Velocity Y")
elif ANGVEL_Z_STATS:
    print("Angular Velocity Z")


# 8) Print out
print(f"Across {len(ns)} runs (total steps = {total_n}):")
print(f"  Pooled Mean        μ = {pooled_mean:.6f}")
print(f"  Pooled StdDev   σ = {pooled_std:.6f}")
print(f"  Overall Maximum   M  = {overall_max:.6f}")
