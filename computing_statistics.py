# Define a statistics function for median and 68% CI
import numpy as np
def calculate_stats(predictions):
    median	= np.median(predictions)
    q1	= np.percentile(predictions, 16)
    q3	= np.percentile(predictions, 84)
    upper_bound = q3 - median
    lower_bound = -1*(median - q1)
    return median, upper_bound, lower_bound
