import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_scatters_with_target(dataframe, target_column="efs", figsize=(16, 12), n_cols=3):
    """
    Create scatter plots for each column against the target column in a single figure.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input data containing features and target column
    target_column : str, default="efs"
        The name of the target column to plot against
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    n_cols : int, default=3
        Number of columns in the subplot grid
    """
    # Filter out the target column from the features
    features = [col for col in dataframe.columns if col != target_column]

    # Calculate number of rows needed
    n_rows = int(np.ceil(len(features) / n_cols))

    # Create a figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'Scatter Plots of Features vs {target_column}', fontsize=16)

    # Flatten the axes array for easy iteration
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # Plot each feature against the target
    for i, feature in enumerate(features):
        if i < len(axes):
            # Create scatter plot
            sns.scatterplot(x=feature, y=target_column, data=dataframe, ax=axes[i], alpha=0.6)

            # Add a trend line
            sns.regplot(x=feature, y=target_column, data=dataframe, ax=axes[i],
                        scatter=False, line_kws={"color": "red"})

            # Set title and labels
            axes[i].set_title(f'{feature} vs {target_column}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target_column)

            # Add correlation value to the plot
            corr = dataframe[feature].corr(dataframe[target_column])
            axes[i].annotate(f'corr = {corr:.2f}',
                             xy=(0.05, 0.95),
                             xycoords='axes fraction',
                             ha='left',
                             va='top',
                             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the overall title

    return fig


# Example usage
if __name__ == "__main__":
    columns = ['dri_score', 'psych_disturb', 'efs']
    columns = None
    drop_columns = ['efs_time', 'ID']
    df = pd.read_csv('data/equity-post-HCT-survival-predictions/train.csv', usecols=columns)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    # Display the dataframe
    df.head()
    # drop first column and get distinct values from each column

    # Get distinct values from each column
    distinct_values = {col: df[col].unique() for col in df.columns}

    # Print distinct values
    for col, values in distinct_values.items():
        print(f"Distinct values in column '{col}': {values}")
    # Map each value to a number (0 to n-1) with n being the number of distinct values order alphabetically and nan last number
    distinct_values_sorted_lambda = lambda col_name: sorted(distinct_values[col_name],
                                                            key=lambda x: (pd.isna(x), str(x).upper()))
    values_map_lambda = lambda col_name: {val: i for i, val in enumerate(distinct_values_sorted_lambda(col_name))}
    df_mapped = df.apply(lambda x: x.map(values_map_lambda(x.name)))
    # get the first 7 columns and last 1
    df_mapped= df_mapped.loc[:, df_mapped.columns[45:] ]

    print(df_mapped.head())


    figure=plot_scatters_with_target(df_mapped)
    #plot the figure
    plt.show()


