from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from os  import path


def stat_test(save_dir):
    # Load the data
    # data = pd.read_csv('./update_figure6/0430/cluster_average.txt', delimiter='\t')
    data_path = path.join(save_dir, 'data/scale_cluster_average.txt')
    data = pd.read_csv(data_path, delimiter='\t', index_col=0)

    # Split the data into before and after treatment
    before_data = data.iloc[:700]
    after_data = data.iloc[700:]

    # Subsample the after_data to match the length of before_data
    after_data_sub = after_data.sample(n=len(before_data), replace=False, random_state=42)

    # Lists to store p-values
    p_values_t = []
    p_values_r = []
    p_values_u = []

    # Perform the statistical tests for each column
    for column in data.columns:
        # Paired t-test
        stat, p = ttest_rel(before_data[column], after_data_sub[column])
        p_values_t.append(p)
        # Wilcoxon signed-rank test
        stat, p = wilcoxon(before_data[column], after_data_sub[column])
        p_values_r.append(p)
        # Mann-Whitney U test
        stat, p = mannwhitneyu(before_data[column], after_data_sub[column])
        p_values_u.append(p)

    # Create a DataFrame to store the p-values
    df = pd.DataFrame({
        'ROI': range(1, data.shape[1] + 1),
        # 'p_values_t': p_values_t,
        'p_values_r': p_values_r,
        # 'p_values_u': p_values_u,
    })

    # Multiple testing correction
    alpha = 0.05

    # No correction
    # df['p_values_t'] = multipletests(df['p_values_t'], alpha=alpha)[1]
    df['p_values_r'] = multipletests(df['p_values_r'], alpha=alpha)[1]
    # df['p_values_u'] = multipletests(df['p_values_u'], alpha=alpha)[1]

    # Bonferroni correction
    # df['p_values_t_bonf'] = multipletests(df['p_values_t'], alpha=alpha, method='bonferroni')[1]
    df['p_values_r_bonf'] = multipletests(df['p_values_r'], alpha=alpha, method='bonferroni')[1]
    # df['p_values_u_bonf'] = multipletests(df['p_values_u'], alpha=alpha, method='bonferroni')[1]

    # Benjamini-Hochberg correction
    # df['p_values_t_bh'] = multipletests(df['p_values_t'], alpha=alpha, method='fdr_bh')[1]
    df['p_values_r_bh'] = multipletests(df['p_values_r'], alpha=alpha, method='fdr_bh')[1]
    # df['p_values_u_bh'] = multipletests(df['p_values_u'], alpha=alpha, method='fdr_bh')[1]

    # Add columns indicating whether the p-value is significant after correction
    df['result_r'] = df['p_values_r'] < alpha
    # df['result_t_bonf'] = df['p_values_t_bonf'] < alpha
    df['result_r_bonf'] = df['p_values_r_bonf'] < alpha
    # df['result_u_bonf'] = df['p_values_u_bonf'] < alpha
    # df['result_t_bh'] = df['p_values_t_bh'] < alpha
    df['result_r_bh'] = df['p_values_r_bh'] < alpha
    # df['result_u_bh'] = df['p_values_u_bh'] < alpha

    # Print the DataFrame
    # print(df)

    # Save the DataFrame to a csv file
    file_path = path.join(save_dir, 'test/test_output.csv')
    df.to_csv(file_path, index=False)
    # with open(file_path, 'w') as f:
    #     f.write(df.to_string())

    before_data_mean = before_data.mean()
    after_data_mean = after_data.mean()

    # Calculate fold change and log2 fold change
    fold_change = after_data_mean / before_data_mean
    # Replace 0 or negative fold changes with np.nan to avoid invalid values in log2
    fold_change_new = np.where(fold_change <= 0, np.nan, fold_change)
    log2_fold_change = np.log2(fold_change_new)

    # Create a DataFrame to store the fold changes
    results = pd.DataFrame({
        'before_treatment': before_data_mean,
        'after_treatment': after_data_mean,
        'Fold Change': fold_change,
        'Log2 Fold Change': log2_fold_change
    })

    # print(results)

    # Save the results to a text file
    fold_change_path = path.join(save_dir, 'test/fold_change_results.txt')
    with open(fold_change_path, 'w') as f:
        f.write(results.to_string())

    plt.figure(figsize=(12, 6))

    # Plotting fold change
    plt.subplot(1, 2, 1)
    results['Fold Change'].plot(kind='bar', color='skyblue')
    plt.xlabel('Parameter')
    plt.ylabel('Fold Change')
    plt.title('Fold Change (After/Before Treatment)')

    # Plotting log2 fold change
    plt.subplot(1, 2, 2)
    results['Log2 Fold Change'].plot(kind='bar', color='salmon')
    plt.xlabel('Parameter')
    plt.ylabel('Log2 Fold Change')
    plt.title('Log2 Fold Change (After/Before Treatment)')

    plt.tight_layout()
    fig_fc = path.join(save_dir, 'test/fold_change.png')
    plt.savefig(fig_fc)
    # plt.show()
    # plt.close()

    new_labels = [label.replace('ROI_', '') for label in results.index]

    # Plot treatment
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(results['before_treatment']))

    bars1 = ax.bar(index, results['before_treatment'], bar_width, label='Pre-treatment', color='blue')
    bars2 = ax.bar(index + bar_width, results['after_treatment'], bar_width, label='Post-treatment', color='red')

    # print(df["p_values_r"])
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        x = bar1.get_x() + bar1.get_width() / 2 + bar_width / 2
        y = max(bar1.get_height(), bar2.get_height()) + 0.01
        ax.text(x, y, f'{df["p_values_r"].iloc[i]:.2e}', ha='center', va='center', color='black')

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average Height')
    ax.set_title('Average Cluster Heights Before and After Treatment')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(new_labels)
    ax.legend()
    fig_treatment = path.join(save_dir, 'test/treatment.png')
    plt.savefig(fig_treatment)
    # plt.show()
    # plt.close()
