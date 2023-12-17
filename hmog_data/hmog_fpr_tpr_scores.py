import numpy as np
import matplotlib.pyplot as plt

# Load the fpr and tpr scores for SSCL model on HMOG dataset

class Scores:
    def __init__(self):
        super().__init__()

        
        # 1) Scores on STD = 1
        fpr_std_1_load = np.genfromtxt('fpr_STD_1.csv', delimiter=',')
        tpr_std_1_load = np.genfromtxt('tpr_STD_1.csv', delimiter=',')

        # Load the fpr and tpr scores for SSCL model on HMOG dataset

        # 2) Scores on STD = 2
        fpr_std_2_load = np.genfromtxt('fpr_STD_2.csv', delimiter=',')
        tpr_std_2_load = np.genfromtxt('tpr_STD_2.csv', delimiter=',')

        # Load the fpr and tpr scores for SSCL model on HMOG dataset

        # 3) Scores on STD = 4
        fpr_std_4_load = np.genfromtxt('fpr_STD_4.csv', delimiter=',')
        tpr_std_4_load = np.genfromtxt('tpr_STD_4.csv', delimiter=',')

        # Load the fpr and tpr scores for SSCL model on HMOG dataset

        # 4) Scores on STD = 8
        fpr_std_8_load = np.genfromtxt('fpr_STD_8.csv', delimiter=',')
        tpr_std_8_load = np.genfromtxt('tpr_STD_8.csv', delimiter=',')

        # Load the fpr and tpr scores for SSCL model on HMOG dataset

        # 5) Scores on STD = 16
        fpr_std_16_load = np.genfromtxt('fpr_STD_16.csv', delimiter=',')
        tpr_std_16_load = np.genfromtxt('tpr_STD_16.csv', delimiter=',')

        # Load the fpr and tpr scores for SSCL model on HMOG dataset

        # 6) Scores on STD = 32
        fpr_std_32_load = np.genfromtxt('fpr_STD_32.csv', delimiter=',')
        tpr_std_32_load = np.genfromtxt('tpr_STD_32.csv', delimiter=',')

        # 7) Scores on STD = 1/2
        fpr_std_half_load = np.genfromtxt('fpr_STD_half.csv', delimiter=',')
        tpr_std_half_load = np.genfromtxt('tpr_STD_half.csv', delimiter=',')

        # 8) Scores on STD = 1/4
        fpr_std_fourth_load = np.genfromtxt('fpr_STD_fourth.csv', delimiter=',')
        tpr_std_fourth_load = np.genfromtxt('tpr_STD_fourth.csv', delimiter=',')

        # 6) Scores on STD = 1/8
        fpr_std_eight_load = np.genfromtxt('fpr_STD_eight.csv', delimiter=',')
        tpr_std_eight_load = np.genfromtxt('tpr_STD_eight.csv', delimiter=',')

        # 10) Scores on STD = 1/16
        fpr_std_sixteen_load = np.genfromtxt('fpr_STD_sixteen.csv', delimiter=',')
        tpr_std_sixteen_load = np.genfromtxt('tpr_STD_sixteen.csv', delimiter=',')

        # 11) Scores on STD = 1/32
        fpr_std_thirty_two_load = np.genfromtxt('fpr_STD_thirty_two.csv', delimiter=',')
        tpr_std_thirty_two_load = np.genfromtxt('tpr_STD_thirty_two.csv', delimiter=',')

        self.fpr_std_1_load = fpr_std_1_load
        self.tpr_std_1_load = tpr_std_1_load

        self.fpr_std_2_load = fpr_std_2_load
        self.tpr_std_2_load = tpr_std_2_load

        self.fpr_std_4_load = fpr_std_4_load
        self.tpr_std_4_load = tpr_std_4_load

        self.fpr_std_8_load = fpr_std_8_load
        self.tpr_std_8_load = tpr_std_8_load

        self.fpr_std_16_load = fpr_std_16_load
        self.tpr_std_16_load = tpr_std_16_load

        self.fpr_std_32_load = fpr_std_32_load
        self.tpr_std_32_load = tpr_std_32_load

        self.fpr_std_half_load = fpr_std_half_load
        self.tpr_std_half_load = tpr_std_half_load

        self.fpr_std_fourth_load = fpr_std_fourth_load
        self.tpr_std_fourth_load = tpr_std_fourth_load

        self.fpr_std_eight_load = fpr_std_eight_load
        self.tpr_std_eight_load = tpr_std_eight_load

        self.fpr_std_sixteen_load = fpr_std_sixteen_load
        self.tpr_std_sixteen_load = tpr_std_sixteen_load

        self.fpr_std_sixteen_load = fpr_std_sixteen_load
        self.tpr_std_sixteen_load = tpr_std_sixteen_load

        self.fpr_std_thirty_two_load = fpr_std_thirty_two_load
        self.tpr_std_thirty_two_load = tpr_std_thirty_two_load


    def double_scores(self):
        array_double = [self.fpr_std_1_load,
                        self.tpr_std_1_load,
                        self.fpr_std_2_load,
                        self.tpr_std_2_load,
                        self.fpr_std_4_load,
                        self.tpr_std_4_load,
                        self.fpr_std_8_load,
                        self.tpr_std_8_load,
                        self.fpr_std_16_load,
                        self.tpr_std_16_load,
                        self.fpr_std_32_load,
                        self.tpr_std_32_load]

        return array_double
    
    def half_scores(self):
        array_half = [self.fpr_std_half_load,
                      self.tpr_std_half_load,
                      self.fpr_std_fourth_load,
                      self.tpr_std_fourth_load,
                      self.fpr_std_eight_load,
                      self.tpr_std_eight_load,
                      self.fpr_std_sixteen_load,
                      self.tpr_std_sixteen_load,
                      self.fpr_std_thirty_two_load,
                      self.tpr_std_thirty_two_load]
        return array_half













