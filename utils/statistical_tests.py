import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, sem, t
import ipdb


df = pd.read_csv('/Users/ynffsy/Downloads/dynamics_paper_numerical_support_2.csv')

df_N1 = df[df['Subject'] == 'N1']
df_N2 = df[df['Subject'] == 'N2']
df_N2_MCL = df[(df['Subject'] == 'N2') & (df['Array'] == 'MC-LAT')]
df_N2_MCM = df[(df['Subject'] == 'N2') & (df['Array'] == 'MC-MED')]

peak_time_ballistic = df['Peak Time (Ballistic)']
peak_time_sustained = df['Peak Time (Sustained)']
left_base_time_ballistic = df['Left Base Time (Ballistic)']
left_base_time_sustained = df['Left Base Time (Sustained)']
peak_duration_ballistic = df['Peak Duration (Ballistic)']
peak_duration_sustained = df['Peak Duration (Sustained)']

## Remove rows where peak_duration_ballistic - peak_duration_sustained is 0
session_filter = (peak_duration_ballistic - peak_duration_sustained == 0)
peak_duration_ballistic = peak_duration_ballistic[~session_filter]
peak_duration_sustained = peak_duration_sustained[~session_filter]

peak_time_N1 = np.concatenate((
    df_N1['Peak Time (Ballistic)'], df_N1['Peak Time (Sustained)']))

peak_time_N2 = np.concatenate((
    df_N2['Peak Time (Ballistic)'], df_N2['Peak Time (Sustained)']))

peak_time_N2_CenterStart = np.concatenate((
    df_N2['Peak Time (Ballistic)'][df_N2['Task'] == 'CenterStart'],
    df_N2['Peak Time (Sustained)'][df_N2['Task'] == 'CenterStart']))

peak_time_N2_RadialGrid = np.concatenate((
    df_N2['Peak Time (Ballistic)'][df_N2['Task'] == 'RadialGrid'],
    df_N2['Peak Time (Sustained)'][df_N2['Task'] == 'RadialGrid']))

peak_time_N2_MCL = np.concatenate((
    df_N2_MCL['Peak Time (Ballistic)'], df_N2_MCL['Peak Time (Sustained)']))

peak_time_N2_MCM = np.concatenate((
    df_N2_MCM['Peak Time (Ballistic)'], df_N2_MCM['Peak Time (Sustained)']))

peak_time_array_filter = (peak_time_N2_MCL - peak_time_N2_MCM == 0)
peak_time_N2_MCL = peak_time_N2_MCL[~peak_time_array_filter]
peak_time_N2_MCM = peak_time_N2_MCM[~peak_time_array_filter]

peak_duration_N1 = np.concatenate((
    df_N1['Peak Duration (Ballistic)'], df_N1['Peak Duration (Sustained)']))

peak_duration_N2 = np.concatenate((
    df_N2['Peak Duration (Ballistic)'], df_N2['Peak Duration (Sustained)']))

peak_duration_N2_CenterStart = np.concatenate((
    df_N2['Peak Duration (Ballistic)'][df_N2['Task'] == 'CenterStart'],
    df_N2['Peak Duration (Sustained)'][df_N2['Task'] == 'CenterStart']))

peak_duration_N2_RadialGrid = np.concatenate((
    df_N2['Peak Duration (Ballistic)'][df_N2['Task'] == 'RadialGrid'],
    df_N2['Peak Duration (Sustained)'][df_N2['Task'] == 'RadialGrid']))

peak_duration_N2_MCL = np.concatenate((
    df_N2_MCL['Peak Duration (Ballistic)'], df_N2_MCL['Peak Duration (Sustained)']))

peak_duration_N2_MCM = np.concatenate((
    df_N2_MCM['Peak Duration (Ballistic)'], df_N2_MCM['Peak Duration (Sustained)']))

left_base_time_N1 = np.concatenate((
    df_N1['Left Base Time (Ballistic)'], df_N1['Left Base Time (Sustained)']))

left_base_time_N2 = np.concatenate((
    df_N2['Left Base Time (Ballistic)'], df_N2['Left Base Time (Sustained)']))

left_base_time_N2_CenterStart = np.concatenate((
    df_N2['Left Base Time (Ballistic)'][df_N2['Task'] == 'CenterStart'],
    df_N2['Left Base Time (Sustained)'][df_N2['Task'] == 'CenterStart']))

left_base_time_N2_RadialGrid = np.concatenate((
    df_N2['Left Base Time (Ballistic)'][df_N2['Task'] == 'RadialGrid'],
    df_N2['Left Base Time (Sustained)'][df_N2['Task'] == 'RadialGrid']))

left_base_time_N2_MCL = np.concatenate((
    df_N2_MCL['Left Base Time (Ballistic)'], df_N2_MCL['Left Base Time (Sustained)']))

left_base_time_N2_MCM = np.concatenate((
    df_N2_MCM['Left Base Time (Ballistic)'], df_N2_MCM['Left Base Time (Sustained)']))

left_base_time_filter = (left_base_time_N2_MCL - left_base_time_N2_MCM == 0)
left_base_time_N2_MCL = left_base_time_N2_MCL[~left_base_time_filter]
left_base_time_N2_MCM = left_base_time_N2_MCM[~left_base_time_filter]


# window_config = 'gt_-0.2_fct_0.5_s0.02_gaussian_0.03_10'

# array1 = peak_onset_time_ballistic
# array2 = peak_onset_time_sustained
# Mann-Whitney U Statistic: 158.5
# P-Value: 0.5155115728252921

# array1 = peak_duration_ballistic
# array2 = peak_duration_sustained
# Mann-Whitney U Statistic: 110.5
# P-Value: 0.03686203562742926

# array1 = peak_onset_time_N1
# array2 = peak_onset_time_N2
# Mann-Whitney U Statistic: 0.0
# P-Value: 7.474138771090956e-05

# array1 = peak_duration_N1
# array2 = peak_duration_N2
# Mann-Whitney U Statistic: 78.5
# P-Value: 0.48387535453394015

# array1 = peak_onset_time_N2_CenterStart
# array2 = peak_onset_time_N2_RadialGrid
# Mann-Whitney U Statistic: 79.5
# P-Value: 0.4609757532522729

# array1 = peak_duration_N2_CenterStart
# array2 = peak_duration_N2_RadialGrid
# Mann-Whitney U Statistic: 110.5
# P-Value: 0.5311042075288077


# window_config = 'gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10'

# array1 = peak_time_ballistic
# array2 = peak_time_sustained
# Test Statistic: 17.5
# P-Value: 0.000789642333984375
# Mean Difference: -0.0233
# 95% CI of Mean Difference: [-0.0409, -0.0057]

# array1 = left_base_time_ballistic
# array2 = left_base_time_sustained
# Test Statistic: 50.5
# P-Value: 0.07283401489257812

# array1 = peak_duration_ballistic
# array2 = peak_duration_sustained
# Test Statistic: 31.5 ## Removing one row where peak_duration_ballistic - peak_duration_sustained is 0
# P-Value: 0.01593017578125
# Mean Difference: -0.0219
# 95% CI of Mean Difference: [-0.0412, -0.0027]

# array1 = peak_time_N1
# array2 = peak_time_N2
# Mann-Whitney U Statistic: 0.0
# P-Value: 0.00013069089168947975

# array1 = left_base_time_N1
# array2 = left_base_time_N2
# Mann-Whitney U Statistic: 0.0
# P-Value: 0.00013002530794947907
# Difference of means: -0.1392
# 95% CI: [-0.1460, -0.1323]

# array1 = peak_duration_N1
# array2 = peak_duration_N2
# Mann-Whitney U Statistic: 69.5
# P-Value: 0.2977415073438059

# array1 = peak_time_N2_CenterStart
# array2 = peak_time_N2_RadialGrid
# Mann-Whitney U Statistic: 56.0
# P-Value: 0.08532445639826056

# array1 = peak_duration_N2_CenterStart
# array2 = peak_duration_N2_RadialGrid
# Mann-Whitney U Statistic: 107.0
# P-Value: 0.6474933305766475

# array1 = peak_time_N2_MCL
# array2 = peak_time_N2_MCM
# Test Statistic: 36.0
# P-Value: 0.1876220703125
# Mean Difference: 0.0073
# 95% CI of Mean Difference: [-0.0138, 0.0285]

array1 = peak_duration_N2_MCL
array2 = peak_duration_N2_MCM

# array1 = left_base_time_N2_MCL
# array2 = left_base_time_N2_MCM
# Test Statistic: 17.0
# P-Value: 0.012451171875
# Mean Difference: 0.0130
# 95% CI of Mean Difference: [0.0035, 0.0225]


# Perform the Wilcoxon signed-rank test
stat, p_value = wilcoxon(array1, array2)
print("Test Statistic:", stat)
print("P-Value:", p_value)


# Perform the Mann-Whitney U test
# stat, p_value = mannwhitneyu(array1, array2, alternative='two-sided')
# print("Mann-Whitney U Statistic:", stat)
# print("P-Value:", p_value)


## Compute 95% CI of mean difference for paired data
differences = array1 - array2
mean_diff = np.mean(differences)
sem_diff = sem(differences)  # = np.std(differences, ddof=1) / np.sqrt(len(differences))

# 4. Compute the t-critical value for a 95% CI with (n-1) degrees of freedom
df = len(differences) - 1
t_crit = t.ppf(1 - 0.025, df)  # two-tailed => alpha/2

# 5. Calculate the margin of error and the confidence interval
margin_of_error = t_crit * sem_diff
ci_lower = mean_diff - margin_of_error
ci_upper = mean_diff + margin_of_error

print(f"Mean Difference: {mean_diff:.4f}")
print(f"95% CI of Mean Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")


## Compute 95% CI of mean difference for unpaired data
# 1. Compute sample means
# mean1 = np.mean(array1)
# mean2 = np.mean(array2)
# mean_diff = mean1 - mean2

# # 2. Compute sample variances
# var1 = np.var(array1, ddof=1)
# var2 = np.var(array2, ddof=1)

# n1 = len(array1)
# n2 = len(array2)

# # 3. Standard error (Welch's formula)
# se_diff = np.sqrt(var1/n1 + var2/n2)

# # 4. Degrees of freedom (Welch–Satterthwaite equation)
# df_numer = (var1/n1 + var2/n2)**2
# df_denom = (var1/n1)**2 / (n1 - 1) + (var2/n2)**2 / (n2 - 1)
# df = df_numer / df_denom  # effective degrees of freedom

# # 5. Critical t-value for a 95% CI
# alpha = 0.05
# t_crit = t.ppf(1 - alpha/2, df)

# # 6. Margin of error
# margin = t_crit * se_diff

# # 7. Confidence interval
# ci_lower = mean_diff - margin
# ci_upper = mean_diff + margin

# print(f"Difference of means: {mean_diff:.4f}")
# print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
