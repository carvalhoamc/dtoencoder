import category_encoders as ce

encoders = {'BackwardDifference': ce.BackwardDifferenceEncoder(),
            'BaseNEncoder': ce.BaseNEncoder(),
            'BinaryEncoder': ce.BinaryEncoder(),
            'CatBoostEncoder': ce.CatBoostEncoder(),
            'CountEncoder': ce.CountEncoder(),
            'GLMMEncoder': ce.GLMMEncoder(),
            'HashingEncoder': ce.HashingEncoder(),
            'HelmertEncoder': ce.HelmertEncoder(),
            'JamesSteinEncoder': ce.JamesSteinEncoder(),
            'LeaveOneOutEncoder': ce.LeaveOneOutEncoder(),
            'MEstimateEncoder': ce.MEstimateEncoder(),
            'OneHotEncoder': ce.OneHotEncoder(),
            'OrdinalEncoder': ce.OrdinalEncoder(),
            'SumEncoder': ce.SumEncoder(),
            'PolynomialEncoder': ce.PolynomialEncoder(),
            'TargetEncoder': ce.TargetEncoder(),
            'WOEEncoder': ce.WOEEncoder()}
