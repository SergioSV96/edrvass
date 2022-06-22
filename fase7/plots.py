import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA



def plot_explained_variance(dataset, pca_model):
    '''Plot the explained variance of all the principal components.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataframe with the data.
    pca_model : sklearn.decomposition.PCA
        The PCA model.
    
    Returns
    ----------
    fig : matplotlib.figure.Figure
        The figure with the plot.
    ax : matplotlib.axes._axes.Axes
        The axes with the plot.
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 25))
    ax.bar(
        x      = np.arange(pca_model.n_components_) + 1,
        height = pca_model.explained_variance_ratio_
    )

    for x, y in zip(np.arange(len(dataset.columns)) + 1, pca_model.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )

    ax.set_xticks(np.arange(pca_model.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Explained variance percentage per principal component')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Percent. explained variance')
    plt.show()

    return fig, ax


def plot_explained_variance_ratio(dataset, pca_model):
    '''Plot the explained variance ratio of all the principal components.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataframe with the data.
    pca_model : sklearn.decomposition.PCA
        The PCA model.
    
    Returns
    ----------
    fig : matplotlib.figure.Figure
        The figure with the plot.
    ax : matplotlib.axes._axes.Axes
        The axes with the plot.
    '''
    prop_varianza_acum = pca_model.explained_variance_ratio_.cumsum()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 25))
    ax.plot(
        np.arange(len(dataset.columns)) + 1,
        prop_varianza_acum,
        marker = 'o'
    )

    for x, y in zip(np.arange(len(dataset.columns)) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
        
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(pca_model.n_components_) + 1)
    ax.set_title('Cumulative explained variance percentage per principal component')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('percent. cumulative explained variance')
    plt.show()

    return fig, ax


def plot_variables_importance(dataset, pca_model):
    '''Plot the variable importance of all the principal components.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataframe with the data.
    pca_model : sklearn.decomposition.PCA
        The PCA model.
    
    Returns
    ----------
    fig : matplotlib.figure.Figure
        The figure with the plot.
    ax : matplotlib.axes._axes.Axes
        The axes with the plot.
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 25))
    componentes = pca_model.components_
    plt.imshow(componentes.T, cmap='magma', aspect='auto')
    plt.yticks(range(len(dataset.columns)), dataset.columns)
    plt.xticks(range(len(dataset.columns)), np.arange(pca_model.n_components_) + 1)
    plt.grid(False)
    plt.colorbar()
    plt.show()

    return fig, ax