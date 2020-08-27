import Orange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classifiers import classifiers_list
from datasets import datasets
from oversampling import order, alphas
from collections import Counter

output_dir = './../output/'
rank_dir = './../rank/'
input_dir = './../input/'

encoders = ['BaseNEncoder',
            'BinaryEncoder',
            'CatBoostEncoder',
            'GLMMEncoder',
            'HashingEncoder',
            'HelmertEncoder',
            'JamesSteinEncoder',
            'LeaveOneOutEncoder',
            'MEstimateEncoder',
            'OneHotEncoder',
            'OrdinalEncoder',
            'SumEncoder',
            'TargetEncoder'
            ]


def split_encoders(results):
	df = pd.read_csv(results)
	dforiginal = df[df['PREPROC'] == 'original']
	dfnc = df[df['PREPROC'] == 'smotenc']
	
	for enc in encoders:
		df1 = df[df['ENCODER'] == enc]
		dfoutput = pd.concat([dforiginal, dfnc, df1])
		dfoutput.to_csv('./../input/' + enc + '.csv', index=False)


class Performance:
	
	def __init__(self):
		pass
	
	def average_results(self, rfile, release):
		'''
		Calculates average results
		:param rfile: filename with results
		:return: avarege_results in another file
		'''
		
		df = pd.read_csv(rfile)
		t = pd.Series(data=np.arange(0, df.shape[0], 1))
		dfr = pd.DataFrame(columns=['ENCODER', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER',
		                            'ALPHA', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA'],
		                   index=np.arange(0, int(t.shape[0] / 5)))
		
		df_temp = df.groupby(by=['ENCODER', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER', 'ALPHA'])
		idx = dfr.index.values
		i = idx[0]
		for name, group in df_temp:
			group = group.reset_index()
			dfr.at[i, 'ENCODER'] = group.loc[0, 'ENCODER']
			dfr.at[i, 'DATASET'] = group.loc[0, 'DATASET']
			dfr.at[i, 'PREPROC'] = group.loc[0, 'PREPROC']
			dfr.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
			dfr.at[i, 'ORDER'] = group.loc[0, 'ORDER']
			dfr.at[i, 'ALPHA'] = group.loc[0, 'ALPHA']
			dfr.at[i, 'PRE'] = group['PRE'].mean()
			dfr.at[i, 'REC'] = group['REC'].mean()
			dfr.at[i, 'SPE'] = group['SPE'].mean()
			dfr.at[i, 'F1'] = group['F1'].mean()
			dfr.at[i, 'GEO'] = group['GEO'].mean()
			dfr.at[i, 'IBA'] = group['IBA'].mean()
			i = i + 1
		
		print('Total lines in a file: ', i)
		dfr.to_csv(input_dir + 'dto_encoders_average_results_' + str(release) + '.csv', index=False)
	
	def rank_by_algorithm(self, df, order, alpha, release, encoder, smote=False):
		'''
		Calcula rank
		:param df:
		:param tipo:
		:param wd:
		:param delaunay_type:
		:return:
		'''
		# df.to_csv('./../output/group/group_'+ encoder+'_'+'.csv')
		measures = ['PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA']
		
		df_table = pd.DataFrame(
				columns=['ENCODER', 'DATASET', 'ALGORITHM', 'ORIGINAL', 'RANK_ORIGINAL', 'SMOTENC',
				         'RANK_SMOTENC', 'SMOTE', 'RANK_SMOTE', 'SMOTE_SVM', 'RANK_SMOTE_SVM', 'BORDERLINE1',
				         'RANK_BORDERLINE1', 'BORDERLINE2', 'RANK_BORDERLINE2', 'GEOMETRIC_SMOTE',
				         'RANK_GEOMETRIC_SMOTE', 'DTO', 'RANK_DTO', 'ORDER', 'ALPHA', 'UNIT'])
		
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			if smote == False:
				df.to_csv(rank_dir + release + '_' + order + '_' + str(alpha) + '_' + encoder + '.csv', index=False)
			# else:
			#	df.to_csv(rank_dir + release + '_smote_' + kind + '_' + order + '_' + str(alpha) + '.csv', index=False)
			
			j = 0
			for d in datasets:
				for m in measures:
					aux = group[group['DATASET'] == d]
					aux = aux.reset_index()
					# aux.to_csv('./../output/group/'+name+str(j)+d+m+'.csv')
					df_table.at[j, 'DATASET'] = d
					df_table.at[j, 'ALGORITHM'] = name
					df_table.at[j, 'ENCODER'] = encoder
					indice = aux.PREPROC[aux.PREPROC == 'original'].index.tolist()[0]
					df_table.at[j, 'ORIGINAL'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == 'smotenc'].index.tolist()[0]
					df_table.at[j, 'SMOTENC'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == 'smote'].index.tolist()[0]
					df_table.at[j, 'SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == 'smoteSVM'].index.tolist()[0]
					df_table.at[j, 'SMOTE_SVM'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == 'borderline1'].index.tolist()[0]
					df_table.at[j, 'BORDERLINE1'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == 'borderline2'].index.tolist()[0]
					df_table.at[j, 'BORDERLINE2'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == 'geometric_smote'].index.tolist()[0]
					df_table.at[j, 'GEOMETRIC_SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.ORDER == order].index.tolist()[0]
					df_table.at[j, 'DTO'] = aux.at[indice, m]
					df_table.at[j, 'ORDER'] = order
					df_table.at[j, 'ALPHA'] = alpha
					df_table.at[j, 'UNIT'] = m
					j += 1
			
			df_pre = df_table[df_table['UNIT'] == 'PRE']
			df_rec = df_table[df_table['UNIT'] == 'REC']
			df_spe = df_table[df_table['UNIT'] == 'SPE']
			df_f1 = df_table[df_table['UNIT'] == 'F1']
			df_geo = df_table[df_table['UNIT'] == 'GEO']
			df_iba = df_table[df_table['UNIT'] == 'IBA']
			
			pre = df_pre[
				['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				 'GEOMETRIC_SMOTE', 'DTO']]
			rec = df_rec[
				['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				 'GEOMETRIC_SMOTE', 'DTO']]
			spe = df_spe[
				['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				 'GEOMETRIC_SMOTE', 'DTO']]
			f1 = df_f1[
				['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				 'GEOMETRIC_SMOTE', 'DTO']]
			geo = df_geo[
				['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				 'GEOMETRIC_SMOTE', 'DTO']]
			iba = df_iba[
				['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				 'GEOMETRIC_SMOTE', 'DTO']]
			
			pre = pre.reset_index()
			pre.drop('index', axis=1, inplace=True)
			rec = rec.reset_index()
			rec.drop('index', axis=1, inplace=True)
			spe = spe.reset_index()
			spe.drop('index', axis=1, inplace=True)
			f1 = f1.reset_index()
			f1.drop('index', axis=1, inplace=True)
			geo = geo.reset_index()
			geo.drop('index', axis=1, inplace=True)
			iba = iba.reset_index()
			iba.drop('index', axis=1, inplace=True)
			
			# calcula rank linha a linha
			pre_rank = pre.rank(axis=1, ascending=False)
			rec_rank = rec.rank(axis=1, ascending=False)
			spe_rank = spe.rank(axis=1, ascending=False)
			f1_rank = f1.rank(axis=1, ascending=False)
			geo_rank = geo.rank(axis=1, ascending=False)
			iba_rank = iba.rank(axis=1, ascending=False)
			
			df_pre = df_pre.reset_index()
			df_pre.drop('index', axis=1, inplace=True)
			df_pre['RANK_ORIGINAL'] = pre_rank['ORIGINAL']
			df_pre['RANK_SMOTE'] = pre_rank['SMOTE']
			df_pre['RANK_SMOTENC'] = pre_rank['SMOTENC']
			df_pre['RANK_SMOTE_SVM'] = pre_rank['SMOTE_SVM']
			df_pre['RANK_BORDERLINE1'] = pre_rank['BORDERLINE1']
			df_pre['RANK_BORDERLINE2'] = pre_rank['BORDERLINE2']
			df_pre['RANK_GEOMETRIC_SMOTE'] = pre_rank['GEOMETRIC_SMOTE']
			df_pre['RANK_DTO'] = pre_rank['DTO']
			
			df_rec = df_rec.reset_index()
			df_rec.drop('index', axis=1, inplace=True)
			df_rec['RANK_ORIGINAL'] = rec_rank['ORIGINAL']
			df_rec['RANK_SMOTENC'] = rec_rank['SMOTENC']
			df_rec['RANK_SMOTE'] = rec_rank['SMOTE']
			df_rec['RANK_SMOTE_SVM'] = rec_rank['SMOTE_SVM']
			df_rec['RANK_BORDERLINE1'] = rec_rank['BORDERLINE1']
			df_rec['RANK_BORDERLINE2'] = rec_rank['BORDERLINE2']
			df_rec['RANK_GEOMETRIC_SMOTE'] = rec_rank['GEOMETRIC_SMOTE']
			df_rec['RANK_DTO'] = rec_rank['DTO']
			
			df_spe = df_spe.reset_index()
			df_spe.drop('index', axis=1, inplace=True)
			df_spe['RANK_ORIGINAL'] = spe_rank['ORIGINAL']
			df_spe['RANK_SMOTENC'] = spe_rank['SMOTENC']
			df_spe['RANK_SMOTE'] = spe_rank['SMOTE']
			df_spe['RANK_SMOTE_SVM'] = spe_rank['SMOTE_SVM']
			df_spe['RANK_BORDERLINE1'] = spe_rank['BORDERLINE1']
			df_spe['RANK_BORDERLINE2'] = spe_rank['BORDERLINE2']
			df_spe['RANK_GEOMETRIC_SMOTE'] = spe_rank['GEOMETRIC_SMOTE']
			df_spe['RANK_DTO'] = spe_rank['DTO']
			
			df_f1 = df_f1.reset_index()
			df_f1.drop('index', axis=1, inplace=True)
			df_f1['RANK_ORIGINAL'] = f1_rank['ORIGINAL']
			df_f1['RANK_SMOTENC'] = f1_rank['SMOTENC']
			df_f1['RANK_SMOTE'] = f1_rank['SMOTE']
			df_f1['RANK_SMOTE_SVM'] = f1_rank['SMOTE_SVM']
			df_f1['RANK_BORDERLINE1'] = f1_rank['BORDERLINE1']
			df_f1['RANK_BORDERLINE2'] = f1_rank['BORDERLINE2']
			df_f1['RANK_GEOMETRIC_SMOTE'] = f1_rank['GEOMETRIC_SMOTE']
			df_f1['RANK_DTO'] = f1_rank['DTO']
			
			df_geo = df_geo.reset_index()
			df_geo.drop('index', axis=1, inplace=True)
			df_geo['RANK_ORIGINAL'] = geo_rank['ORIGINAL']
			df_geo['RANK_SMOTENC'] = geo_rank['SMOTENC']
			df_geo['RANK_SMOTE'] = geo_rank['SMOTE']
			df_geo['RANK_SMOTE_SVM'] = geo_rank['SMOTE_SVM']
			df_geo['RANK_BORDERLINE1'] = geo_rank['BORDERLINE1']
			df_geo['RANK_BORDERLINE2'] = geo_rank['BORDERLINE2']
			df_geo['RANK_GEOMETRIC_SMOTE'] = geo_rank['GEOMETRIC_SMOTE']
			df_geo['RANK_DTO'] = geo_rank['DTO']
			
			df_iba = df_iba.reset_index()
			df_iba.drop('index', axis=1, inplace=True)
			df_iba['RANK_ORIGINAL'] = iba_rank['ORIGINAL']
			df_iba['RANK_SMOTENC'] = iba_rank['SMOTENC']
			df_iba['RANK_SMOTE'] = iba_rank['SMOTE']
			df_iba['RANK_SMOTE_SVM'] = iba_rank['SMOTE_SVM']
			df_iba['RANK_BORDERLINE1'] = iba_rank['BORDERLINE1']
			df_iba['RANK_BORDERLINE2'] = iba_rank['BORDERLINE2']
			df_iba['RANK_GEOMETRIC_SMOTE'] = iba_rank['GEOMETRIC_SMOTE']
			df_iba['RANK_DTO'] = iba_rank['DTO']
			
			# avarege rank
			media_pre_rank = pre_rank.mean(axis=0)
			media_rec_rank = rec_rank.mean(axis=0)
			media_spe_rank = spe_rank.mean(axis=0)
			media_f1_rank = f1_rank.mean(axis=0)
			media_geo_rank = geo_rank.mean(axis=0)
			media_iba_rank = iba_rank.mean(axis=0)
			
			media_pre_rank_file = media_pre_rank.reset_index()
			media_pre_rank_file = media_pre_rank_file.sort_values(by=0)
			
			media_rec_rank_file = media_rec_rank.reset_index()
			media_rec_rank_file = media_rec_rank_file.sort_values(by=0)
			
			media_spe_rank_file = media_spe_rank.reset_index()
			media_spe_rank_file = media_spe_rank_file.sort_values(by=0)
			
			media_f1_rank_file = media_f1_rank.reset_index()
			media_f1_rank_file = media_f1_rank_file.sort_values(by=0)
			
			media_geo_rank_file = media_geo_rank.reset_index()
			media_geo_rank_file = media_geo_rank_file.sort_values(by=0)
			
			media_iba_rank_file = media_iba_rank.reset_index()
			media_iba_rank_file = media_iba_rank_file.sort_values(by=0)
			
			if smote == False:
				# Grava arquivos importantes
				df_pre.to_csv(
						rank_dir + release + '_' + encoder + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_pre.csv', index=False)
				df_rec.to_csv(
						rank_dir + release + '_' + encoder + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_rec.csv', index=False)
				df_spe.to_csv(
						rank_dir + release + '_' + encoder + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_spe.csv', index=False)
				df_f1.to_csv(
						rank_dir + release + '_' + encoder + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_f1.csv', index=False)
				df_geo.to_csv(
						rank_dir + release + '_' + encoder + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_geo.csv', index=False)
				df_iba.to_csv(
						rank_dir + release + '_' + encoder + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_iba.csv', index=False)
				
				media_pre_rank_file.to_csv(
						rank_dir + release + '_' + encoder + '_' + 'media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_pre.csv',
						index=False)
				media_rec_rank_file.to_csv(
						rank_dir + release + '_' + encoder + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_rec.csv',
						index=False)
				media_spe_rank_file.to_csv(
						rank_dir + release + '_' + encoder + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_spe.csv',
						index=False)
				media_f1_rank_file.to_csv(
						rank_dir + release + '_' + encoder + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_f1.csv',
						index=False)
				media_geo_rank_file.to_csv(
						rank_dir + release + '_' + encoder + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_geo.csv',
						index=False)
				media_iba_rank_file.to_csv(
						rank_dir + release + '_' + encoder + '_media_rank_' + order + '_' + str(
								alpha) + '_' + name + '_iba.csv',
						index=False)
				
				delaunay_type = order + '_' + str(alpha)
				
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTENC', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2',
				                   'GEOMETRIC_SMOTE', 'dto_' + delaunay_type]
				avranks = list(media_pre_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + '_' + encoder + 'cd_' + delaunay_type + '_' + name + '_pre.pdf')
				plt.close()
				
				avranks = list(media_rec_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + '_' + encoder + 'cd_' + delaunay_type + '_' + name + '_rec.pdf')
				plt.close()
				
				avranks = list(media_spe_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + '_' + encoder + 'cd_' + delaunay_type + '_' + name + '_spe.pdf')
				plt.close()
				
				avranks = list(media_f1_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(rank_dir + release + '_' + encoder + 'cd_' + delaunay_type + '_' + name + '_f1.pdf')
				plt.close()
				
				avranks = list(media_geo_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + '_' + encoder + 'cd_' + delaunay_type + '_' + name + '_geo.pdf')
				plt.close()
				
				avranks = list(media_iba_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(datasets))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + '_' + encoder + 'cd_' + delaunay_type + '_' + name + '_iba.pdf')
				plt.close()
				
				print('DTO = ', delaunay_type)
				print('Algorithm= ', name)
	
	def rank_dto_by(self, order, alpha, release, encoder, smote=False):
		M = ['_pre.csv', '_rec.csv', '_spe.csv', '_f1.csv', '_geo.csv', '_iba.csv']
		
		df_media_rank = pd.DataFrame(columns=['ENCODER', 'ALGORITHM', 'RANK_ORIGINAL', 'RANK_SMOTENC', 'RANK_SMOTE',
		                                      'RANK_SMOTE_SVM', 'RANK_BORDERLINE1', 'RANK_BORDERLINE2',
		                                      'RANK_GEOMETRIC_SMOTE', 'RANK_DTO', 'UNIT'])
		geometry = order + '_' + alpha
		if smote == False:
			name = rank_dir + release + '_' + encoder + '_total_rank_' + geometry + '_'
		
		for m in M:
			i = 0
			for c in classifiers_list:
				df = pd.read_csv(name + c + m)
				rank_original = df.RANK_ORIGINAL.mean()
				rank_smotenc = df.RANK_SMOTENC.mean()
				rank_smote = df.RANK_SMOTE.mean()
				rank_smote_svm = df.RANK_SMOTE_SVM.mean()
				rank_b1 = df.RANK_BORDERLINE1.mean()
				rank_b2 = df.RANK_BORDERLINE2.mean()
				rank_geo_smote = df.RANK_GEOMETRIC_SMOTE.mean()
				rank_dto = df.RANK_DTO.mean()
				df_media_rank.loc[i, 'ENCODER'] = encoder
				df_media_rank.loc[i, 'ALGORITHM'] = df.loc[0, 'ALGORITHM']
				df_media_rank.loc[i, 'RANK_ORIGINAL'] = rank_original
				df_media_rank.loc[i, 'RANK_SMOTENC'] = rank_smotenc
				df_media_rank.loc[i, 'RANK_SMOTE'] = rank_smote
				df_media_rank.loc[i, 'RANK_SMOTE_SVM'] = rank_smote_svm
				df_media_rank.loc[i, 'RANK_BORDERLINE1'] = rank_b1
				df_media_rank.loc[i, 'RANK_BORDERLINE2'] = rank_b2
				df_media_rank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = rank_geo_smote
				df_media_rank.loc[i, 'RANK_DTO'] = rank_dto
				df_media_rank.loc[i, 'UNIT'] = df.loc[0, 'UNIT']
				i += 1
			
			dfmediarank = df_media_rank.copy()
			dfmediarank = dfmediarank.sort_values('RANK_DTO')
			
			dfmediarank.loc[i, 'ALGORITHM'] = 'avarage'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].mean()
			dfmediarank.loc[i, 'RANK_SMOTENC'] = df_media_rank['RANK_SMOTENC'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].mean()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_DTO'] = df_media_rank['RANK_DTO'].mean()
			dfmediarank.loc[i, 'UNIT'] = df.loc[0, 'UNIT']
			i += 1
			dfmediarank.loc[i, 'ALGORITHM'] = 'std'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].std()
			dfmediarank.loc[i, 'RANK_SMOTENC'] = df_media_rank['RANK_SMOTENC'].std()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].std()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_DTO'] = df_media_rank['RANK_DTO'].std()
			dfmediarank.loc[i, 'UNIT'] = df.loc[0, 'UNIT']
			
			dfmediarank['RANK_ORIGINAL'] = pd.to_numeric(dfmediarank['RANK_ORIGINAL'], downcast="float").round(2)
			dfmediarank['RANK_SMOTENC'] = pd.to_numeric(dfmediarank['RANK_SMOTENC'], downcast="float").round(2)
			dfmediarank['RANK_SMOTE'] = pd.to_numeric(dfmediarank['RANK_SMOTE'], downcast="float").round(2)
			dfmediarank['RANK_SMOTE_SVM'] = pd.to_numeric(dfmediarank['RANK_SMOTE_SVM'], downcast="float").round(2)
			dfmediarank['RANK_BORDERLINE1'] = pd.to_numeric(dfmediarank['RANK_BORDERLINE1'], downcast="float").round(2)
			dfmediarank['RANK_BORDERLINE2'] = pd.to_numeric(dfmediarank['RANK_BORDERLINE2'], downcast="float").round(2)
			dfmediarank['RANK_GEOMETRIC_SMOTE'] = pd.to_numeric(dfmediarank['RANK_GEOMETRIC_SMOTE'],
			                                                    downcast="float").round(2)
			dfmediarank['RANK_DTO'] = pd.to_numeric(dfmediarank['RANK_DTO'], downcast="float").round(2)
			
			if smote == False:
				dfmediarank.to_csv(output_dir + release + '_' + encoder + '_results_media_rank_' + geometry + m,
				                   index=False)
	
	def grafico_variacao_alpha(self, release):
		M = ['_geo', '_iba']
		
		df_alpha_variations_rank = pd.DataFrame()
		df_alpha_variations_rank['alphas'] = alphas
		df_alpha_variations_rank.index = alphas
		
		df_alpha_all = pd.DataFrame()
		df_alpha_all['alphas'] = alphas
		df_alpha_all.index = alphas
		
		for enc in encoders:
			for m in M:
				for o in order:
					for a in alphas:
						filename = output_dir + release + '_' + enc + '_results_media_rank_' + o + '_' + str(
								a) + m + '.csv'
						print(filename)
						df = pd.read_csv(filename)
						mean = df.loc[8, 'RANK_DTO']
						df_alpha_variations_rank.loc[a, 'AVARAGE_RANK'] = mean
					
					if m == '_geo':
						measure = 'GEO'
					if m == '_iba':
						measure = 'IBA'
					
					df_alpha_all[o + '_' + measure] = df_alpha_variations_rank['AVARAGE_RANK'].copy()
					
					fig, ax = plt.subplots()
					ax.set_title('DTO AVARAGE RANK\n ' + 'GEOMETRY = ' + o + '\nMEASURE = ' + measure, fontsize=10)
					ax.set_xlabel('Alpha')
					ax.set_ylabel('Rank')
					ax.plot(df_alpha_variations_rank['AVARAGE_RANK'], marker='d', label='Avarage Rank')
					ax.legend(loc="upper right")
					plt.xticks(range(11))
					fig.savefig(output_dir + release + '_' + enc + '_pic_' + o + '_' + measure + '.png', dpi=125)
					plt.show()
					plt.close()
			
			# figure(num=None, figsize=(10, 10), dpi=800, facecolor='w', edgecolor='k')
			
			fig, ax = plt.subplots(figsize=(10, 7))
			ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = GEO', fontsize=5)
			ax.set_xlabel('Alpha')
			ax.set_ylabel('Rank')
			t1 = df_alpha_all['alphas']
			t2 = df_alpha_all['alphas']
			ft1 = df_alpha_all['aspect_ratio_GEO']
			ft2 = df_alpha_all['solid_angle_GEO']
			
			ax.plot(t1, ft1, color='tab:blue', marker='o', label='aspect_ratio')
			ax.plot(t2, ft2, color='tab:red', marker='o', label='solid_angle')
			
			leg = ax.legend(loc='upper right')
			leg.get_frame().set_alpha(0.5)
			plt.xticks(range(12))
			plt.savefig(output_dir + release + '_' + enc + '_pic_all_geo.png', dpi=800)
			plt.show()
			plt.close()
			df_alpha_all.to_csv(output_dir + release + '_' + enc + '_pic_all_geo.csv', index=False)
			
			###################
			fig, ax = plt.subplots(figsize=(10, 7))
			ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = IBA', fontsize=5)
			ax.set_xlabel('Alpha')
			ax.set_ylabel('Rank')
			t1 = df_alpha_all['alphas']
			t2 = df_alpha_all['alphas']
			
			ft1 = df_alpha_all['aspect_ratio_IBA']
			ft2 = df_alpha_all['solid_angle_IBA']
			
			ax.plot(t1, ft1, color='tab:blue', marker='o', label='aspect_ratio')
			ax.plot(t2, ft2, color='tab:red', marker='o', label='solid_angle')
			
			leg = ax.legend(loc='upper right')
			leg.get_frame().set_alpha(0.5)
			plt.xticks(range(12))
			plt.savefig(output_dir + release + '_' + enc + '_pic_all_iba.png', dpi=800)
			plt.show()
			plt.close()
			df_alpha_all.to_csv(output_dir + release + '_' + enc + '_pic_all_iba.csv', index=False)
	
	def best_alpha_geometry(self):
		# Best alpha calculation
		# TODO
		df = pd.DataFrame(columns=['alphas', 'aspect_ratio_GEO', 'solid_angle_GEO', 'aspect_ratio_IBA',
		                           'solid_angle_IBA', 'ENCODER'])
		for enc in encoders:
			df1 = pd.read_csv(output_dir + 'v1' + '_' + enc + '_pic_all_geo.csv')
			df1['ENCODER'] = enc
			df = pd.concat([df, df1])
		
		M = ['_GEO', '_IBA']
		for o in order:
			for m in M:
				vals = []
				for enc in encoders:
					df_enc = df[df['ENCODER'] == enc]
					vals.append(list(df_enc[o + m]))
				
				ft1 = vals[0]
				ft2 = vals[1]
				ft3 = vals[2]
				ft4 = vals[3]
				ft5 = vals[4]
				ft6 = vals[5]
				ft7 = vals[6]
				ft8 = vals[7]
				ft9 = vals[8]
				ft10 = vals[9]
				ft11 = vals[10]
				ft12 = vals[11]
				ft13 = vals[12]
				
				fig, ax = plt.subplots(figsize=(10, 7))
				
				t1 = alphas
				t2 = alphas
				t3 = alphas
				t4 = alphas
				t5 = alphas
				t6 = alphas
				t7 = alphas
				t8 = alphas
				t9 = alphas
				t10 = alphas
				t11 = alphas
				t12 = alphas
				t13 = alphas
				
				if m == '_GEO':
					ext = 'GEO'
				if m == '_IBA':
					ext = 'IBA'
				
				ax.set_title('DTO RANK( ' + ext + ')\n ' + o + '\n', fontsize=5)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('Rank')
				
				ax.plot(t1, ft1, color='tab:blue', marker='o', label='BaseNEncoder')
				ax.plot(t2, ft2, color='tab:red', marker='o', label='BinaryEncoder')
				ax.plot(t3, ft3, color='tab:green', marker='o', label='CatBoostEncoder')
				ax.plot(t4, ft4, color='tab:orange', marker='o', label='GLMMEncoder')
				ax.plot(t5, ft5, color='tab:olive', marker='o', label='HashingEncoder')
				ax.plot(t6, ft6, color='tab:purple', marker='o', label='HelmertEncoder')
				ax.plot(t7, ft7, color='tab:brown', marker='o', label='JamesSteinEncoder')
				ax.plot(t8, ft8, color='tab:pink', marker='o', label='LeaveOneOutEncoder')
				ax.plot(t9, ft9, color='tab:gray', marker='o', label='MEstimateEncoder')
				ax.plot(t10, ft10, color='tab:purple', marker='x', label='OneHotEncoder')
				ax.plot(t11, ft11, color='tab:brown', marker='x', label='OrdinalEncoder')
				ax.plot(t12, ft12, color='tab:pink', marker='x', label='SumEncoder')
				ax.plot(t13, ft13, color='tab:gray', marker='x', label='TargetEncoder')
				
				leg = ax.legend(loc='upper right')
				leg.get_frame().set_alpha(0.5)
				plt.xticks(range(12))
				if m == '_GEO':
					ext = 'geo'
				if m == '_IBA':
					ext = 'iba'
				plt.savefig(output_dir + o + '_' + '_pic_best_encoder_' + ext + '.png', dpi=800)
				plt.show()
				plt.close()
				df.to_csv(output_dir + o + '_' + '_pic_best_encoder_' + ext + '.csv', index=False)
	
	def run_rank_choose_parameters(self, release):
		for enc in encoders:
			filename = enc + '.csv'
			df_best_dto = pd.read_csv(input_dir + filename)
			df_B1 = df_best_dto[df_best_dto['PREPROC'] == 'borderline1'].copy()
			df_B2 = df_best_dto[df_best_dto['PREPROC'] == 'borderline2'].copy()
			df_GEO = df_best_dto[df_best_dto['PREPROC'] == 'geometric_smote'].copy()
			df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == 'smote'].copy()
			df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == 'smoteSVM'].copy()
			df_original = df_best_dto[df_best_dto['PREPROC'] == 'original'].copy()
			df_nc = df_best_dto[df_best_dto['PREPROC'] == 'smotenc'].copy()
			df_dto = df_best_dto[df_best_dto['PREPROC'] == 'dtosmote'].copy()
			for o in order:
				for a in alphas:
					df_order = df_dto[df_dto['ORDER'] == str(o)].copy()
					print(Counter(df_order['PREPROC']))
					df_alpha = df_order[df_order['ALPHA'] == str(a)].copy()
					print(Counter(df_alpha['PREPROC']))
					df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_alpha, df_original, df_nc])
					print('contador= ', Counter(df['PREPROC']))
					self.rank_by_algorithm(df, o, str(a), release, enc)
					self.rank_dto_by(o, str(a), release, enc)
	
	def run_global_rank(self, filename, kind, release):
		df_best_dto = pd.read_csv(filename)
		df_B1 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline1'].copy()
		df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
		df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
		df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()
		o = 'solid_angle'
		if kind == 'biclass':
			a = 7.0
		else:
			a = 7.5
		
		GEOMETRY = '_delaunay_' + o + '_' + str(a)
		df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
		df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_original, df_dto])
		self.rank_by_algorithm(df, kind, o, str(a), release, smote=True)
		self.rank_dto_by(o + '_' + str(a), kind, release, smote=True)
	
	def overall_rank(self, ext, release, geometry, alpha):
		i = 0
		df_mean = pd.DataFrame()
		for enc in encoders:
			df1 = pd.DataFrame()
			for cls in classifiers_list:
				df = pd.read_csv(
						rank_dir + release + '_' + enc + '_total_rank_' + geometry + '_' + str(
								alpha) + '_' + cls + '_' + ext + '.csv')
				df1 = pd.concat([df1, df])
			
			df1 = df1.reset_index()
			df_mean.loc[i, 'ENCODER'] = df1.loc[0, 'ENCODER']
			df_mean.loc[i, 'UNIT'] = df1.loc[0, 'UNIT']
			df_mean.loc[i, 'ORDER'] = df1.loc[0, 'ORDER']
			df_mean.loc[i, 'ALPHA'] = df1.loc[0, 'ALPHA']
			
			df_mean.loc[i, 'RANK_ORIGINAL'] = pd.to_numeric(df1['RANK_ORIGINAL'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_SMOTENC'] = pd.to_numeric(df1['RANK_SMOTENC'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_SMOTE'] = pd.to_numeric(df1['RANK_SMOTE'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_SMOTE_SVM'] = pd.to_numeric(df1['RANK_SMOTE_SVM'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_BORDERLINE1'] = pd.to_numeric(df1['RANK_BORDERLINE1'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_BORDERLINE2'] = pd.to_numeric(df1['RANK_BORDERLINE2'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_GEOMETRIC_SMOTE'] = pd.to_numeric(df1['RANK_GEOMETRIC_SMOTE'].mean(), downcast="float")
			df_mean.loc[i, 'RANK_DTO'] = pd.to_numeric(df1['RANK_DTO'].mean(), downcast="float")
			i = i + 1
			
		df_mean.to_csv(
					output_dir + 'by_encoders_rank_results_' + geometry + '_' + str(
							alpha) + '_' + ext + '.csv',
					index=False)
		
		for j in np.arange(0,len(encoders)):
			# grafico CD
			identificadores = ['RANK_ORIGINAL','RANK_SMOTENC','RANK_SMOTE','RANK_SMOTE_SVM','RANK_BORDERLINE1',
			                   'RANK_BORDERLINE2','RANK_GEOMETRIC_SMOTE','RANK_DTO']
			media = df_mean.loc[j,:]
			media = media[4:]
			avranks = list(media)
			cd = Orange.evaluation.compute_CD(avranks, len(datasets))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
				output_dir + 'by_encoders_rank_results_'+ df_mean.loc[j,'ENCODER'] + '_cd_' +
				geometry + '_' + str(alpha) + '_' + ext +'.pdf')
		plt.close()
		
		
		
	
	def cd_graphics(self, df, datasetlen, kind):  # TODO
		# grafico CD
		names = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']
		algorithms = classifiers_list
		
		for i in np.arange(0, len(algorithms)):
			avranks = list(df.loc[i])
			algorithm = avranks[0]
			measure = avranks[1]
			avranks = avranks[2:]
			cd = Orange.evaluation.compute_CD(avranks, datasetlen)
			Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=len(algorithms), textspace=3)
			plt.savefig(output_dir + kind + '_cd_' + algorithm + '_' + measure + '.pdf')
			plt.close()
