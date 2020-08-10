from dtosmote import DTO
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from gsmote import GeometricSMOTE
import warnings

oversampling_methods = {'smote': SMOTE(),
                        'borderline1': BorderlineSMOTE(kind='borderline-1'),
                        'borderline2': BorderlineSMOTE(kind='borderline-2'),
                        'smoteSVM': SVMSMOTE(),
                        'geometric_smote': GeometricSMOTE(n_jobs=-1),
                        #'dtosmote': DTO(dataset_name='uname',
                        #                geometry='solid_angle',
                        #                dirichlet=7)
                        }
