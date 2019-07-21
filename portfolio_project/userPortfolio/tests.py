from django.test import TestCase

# Create your tests here.
def run_multiple_ensenble():
    import pickle
    import userPortfolio.ensamble as eble
    period_len = 20
    for i in range(5):
        clc = eble.EnsambleClassifier(num_days=200)
        ret = clc.future_test(test_period=1, num_periods=(i*period_len)+period_len,
        period_start=i*period_len, moveback=2)
        if i == 0:
            data = clc.combined_result
            fp = open('ensemble_result.txt', 'wb')
            pickle.dump(data, fp)
            fp.close()
        else:
            fp = open('ensemble_result.txt', 'rb')
            current_data = pickle.load(fp)
            fp.close()
            for eachdp in clc.combined_result:
                current_data.append(eachdp)
            fp = open('ensemble_result.txt', 'wb')
            pickle.dump(current_data, fp)
        fp.close()
