"""
This Script Applies LIME and SHAP parameter selection (lime_shap_parameter_selection.py) to all 5 datasets when analysis_mode=True.
"""

import sys
sys.path.insert(0, ".")

from tasks.boston_housing import boston_exp
from tasks.communities_crime import cc_exp
from tasks.compas import compas_exp
from tasks.german_credit import german_exp
from tasks.student_performance import student_exp

if __name__ == "__main__":
    boston_exp(analysis_mode=True)
    cc_exp(analysis_mode=True)
    compas_exp(analysis_mode=True)
    german_exp(analysis_mode=True)
    student_exp(analysis_mode=True)

