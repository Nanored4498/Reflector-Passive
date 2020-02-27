import exercises as ex
import os
import warnings
warnings.filterwarnings('ignore')

def setup():
	print("checking directories...\n")
	l = os.listdir()
	questions_to_do = ex.questions_ready
	if 'figs' not in l:
		os.mkdir('figs')
	else:
		l = os.listdir('figs/')
		if "plot tau -> C_N(tau,x_1,x_5).png" in l:
			questions_to_do[0][0] = -1
		if "image KM de I_N.png" in l:
			questions_to_do[0][1] = -1
		if questions_to_do[0][0] == -1 and questions_to_do[0][1] == -1:
			questions_to_do[0][2] = -1
		if 'plot tau -> C_NTM(tau,x_5,x_1) 500.png' in l:
			questions_to_do[1][0] = -1
	return questions_to_do