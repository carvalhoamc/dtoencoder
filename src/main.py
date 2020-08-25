from encode_datasets import run
import warnings
import time

from performance import Performance, input_dir, output_dir, split_encoders

warnings.filterwarnings('ignore')

def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))

def run_experiments():
	start = time.time()
	run(onlyDTO=False)  # if True only DTO runs
	end = time.time()
	print("Total Execution Time : ")
	timer(start, end)

def run_analisys(r):
	analisys = Performance()
	#analisys.average_results(output_dir + 'encoder_results_' + r + '.csv',  release=r)
	analisys.run_rank_choose_parameters(release=r)
	


def main():
	start = time.time()
	#run_experiments()
	#split_encoders('./../input/dto_encoders_average_results_v1.csv')
	run_analisys('v1')
	
	end = time.time()
	print("Total Execution Time : ")
	timer(start, end)
	
	
	
	
if __name__ == '__main__':
    main()