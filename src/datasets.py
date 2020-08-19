datasets = [
		'abalone-3_vs_11',  # ok
		'kddcup-land_vs_portsweep',  # ok
		'dataset_31_credit-g',  # ok
		'dresses-sales',  # ok
		'dermatology',  # ok
		'thyroid-hypothyroid',  # ok
		#'acute-inflammations-nephritis',
		'credit-approval',  # ok
		'horse-colic-surgical',  # ok
		'heart',  # ok
		'hepatitis',  # ok

]

dict_categorical_cols = {'abalone-3_vs_11': [0],
                         'kddcup-land_vs_portsweep': [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                      9, 10, 11, 12, 13, 14, 15,
                                                      16, 17, 18, 19, 20, 21, 23,
                                                      32, ],
                         'heart': [2, 6, 9, 7, 3, 11],
                         'dataset_31_credit-g': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10, 11, 12, 13, 14, 15, 16, 17,
                                                 18, 19],
                         'dresses-sales': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         'hepatitis': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       18],
                         'dermatology': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                         32],
                         'thyroid-hypothyroid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                 11, 12, 13, 15, 17, 19, 21, 23],
                         'acute-inflammations-nephritis': [1, 2, 3, 4, 5],
                         'credit-approval': [0, 2, 3, 4, 5, 7, 8, 9, 10, 11],
                         'horse-colic-surgical': [0, 1, 6, 7, 8, 9, 10, 11, 12,
                                                  13, 14, 16, 17, 20, 22, 24, 25,
                                                  26]
	
                         }
