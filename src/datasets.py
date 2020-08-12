datasets = [
		# 'abalone-3_vs_11',  # ok
		# 'kddcup-land_vs_portsweep',  # ok
		# 'heart' #ok
		'dataset_31_credit-g'  # ok
		
		# 'smofn-3-7-10',
		# 'smux6',
		# 'sthreeOf9',
		# 'sphpV5QYya',
		# 'bands',  #Cylinder Bands Data Set
		# 'echocardiogram',
		# 'pasture',
		# 'white-clover',
		# 'dataset_9_autos',
		# 'kick',
		# 'autos',

]

dict_categorical_cols = {'abalone-3_vs_11': [0],
                         'kddcup-land_vs_portsweep': [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                      9,
                                                      10, 11, 12,
                                                      13, 14, 15, 16, 17, 18,
                                                      19, 20,
                                                      21, 23,
                                                      32, ],
                         'smofn-3-7-10': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         'smux6': [0, 1, 2, 3, 4, 5],
                         'sthreeOf9': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                         'sphpV5QYya': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                        12, 13,
                                        14, 15, 16, 17,
                                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                        28,
                                        29],
                         'bands': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15],
                         'echocardiogram': [0, 1, 2, 3, 5, 7, 8, 9, 10, 11],
                         'pasture': [0, 1, 2, 3, 4, 5, 13, 21],
                         'white-clover': [0, 1, 2, 30],
                         'heart': [2, 6, 9, 7, 3, 11],
                         'kick': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'autos': [1, 2, 3, 4, 5, 6, 7, 13, 14, 16, 21],
                         'dataset_31_credit-g': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10, 11, 12,
                                                 13, 14, 15, 16, 17, 18, 19]
	
                         }
