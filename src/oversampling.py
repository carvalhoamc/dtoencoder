from dtosmote import DTO
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from gsmote import GeometricSMOTE
import numpy as np
import warnings

oversampling_methods = {'smote': SMOTE(),
                        'borderline1': BorderlineSMOTE(kind='borderline-1'),
                        'borderline2': BorderlineSMOTE(kind='borderline-2'),
                        'smoteSVM': SVMSMOTE(),
                        'geometric_smote': GeometricSMOTE(n_jobs=-1),
                        }


#geometry
order = [#'area',#ok
         #'volume',#ok
         #'area_volume_ratio',#ok
         #'edge_ratio',#ok
         #'radius_ratio',#ok
         'aspect_ratio',#ok
         #'max_solid_angle',
         #'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = [0.5,1,7]

