from sklearn.utils.estimator_checks import check_estimator
from pIC50_predictor.sklearn_predictor import pIC50_predictor

def test_estimator():
    check_estimator(pIC50_predictor)

if __name__ == "__main__":
    test_estimator()
