import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



logger = logging.getLogger("reporting")


def loss_vs_removed_count(report_df):
    sns.lineplot(x='removed-count', y='loss', data=report_df)
    plt.title('Loss vs. Number of Features Removed')
    plt.xlabel('Number of Features Removed')
    plt.ylabel('Loss')
    plt.show()

