'''
Compute the error of estimation by the distance metrics mentioned in the article
1. Mean estimation error(Gamma fit/Midpoint/VAM)(Population/Cohort)
2. Probability vector estimation error(Linfinity/KL divergence)(Population/Cohort)
3. Bootstrap analysis 95% CI (.ipynb on healthdb)
'''

import numpy as np 
import pandas as pd 
from config import Config_simul