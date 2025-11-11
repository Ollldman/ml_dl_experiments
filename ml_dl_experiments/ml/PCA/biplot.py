import pandas as pd
import matplotlib.pyplot as plt
import mpl_axes_aligner

def biplot(df_scores: pd.DataFrame, df_loadings: pd.DataFrame) -> None:
    """
    Create a biplot visualization for PCA results.
    
    A biplot displays both the principal component scores (as points) and 
    the principal component loadings (as vectors) on the same plot. This allows
    visualization of both the sample relationships and variable contributions.
    
    Parameters
    ----------
    df_scores : pd.DataFrame
        DataFrame containing principal component scores. Must contain 
        'PC1' and 'PC2' columns for the first two principal components.
    df_loadings : pd.DataFrame
        DataFrame containing principal component loadings. Must contain 
        'PC1' and 'PC2' columns for the first two principal components.
        The index should contain variable names.
    
    Returns
    -------
    None
        Displays the biplot using matplotlib.
    
    Examples
    --------
    >>> biplot(scores_df, loadings_df)
    """
    if 'PC1' in df_scores.columns and 'PC2' in df_scores.columns:
        fig, ax = plt.subplots(figsize=(15, 8))
        # Convert to numpy arrays explicitly to fix type issues
        pc1_scores = df_scores['PC1'].values
        pc2_scores = df_scores['PC2'].values
        ax.scatter(pc1_scores, pc2_scores, c='b')
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)

        # create a second set of axes
        ax2 = ax.twinx().twiny()

        # setup font dictionary:
        font = {
            'color': 'g',
            'weight': 'bold',
            'size' : 12
        }

        for col in df_loadings.columns.values:
            tipx = float(df_loadings.loc['PC1', col])
            tipy = float(df_loadings.loc['PC2', col])
            ax2.arrow(
                0, 
                0, 
                tipx, 
                tipy, 
                color='r', 
                alpha = 0.5,
                head_width=0.05,
                head_length=0.05)
            
            ax2.text(
                tipx*1.05,
                tipy*1.05, 
                col, 
                fontdict=font, 
                ha='center', 
                va='center')
        
        # align x = 0 of ax and ax2 with center of figure:
        mpl_axes_aligner.align.xaxes(ax, 0, ax2, 0, 0.5)
        # align y = 0 of ax and ax2 with center of figure:
        mpl_axes_aligner.align.yaxes(ax, 0, ax2, 0, 0.5)

        plt.show()
    else:
        print("PC1 and PC2 are not in DataFrame columns")