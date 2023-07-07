# NFL Score Prediction Model 2019 Season


Creators: Dennis Nazarov, Daniel Zafman

# Introduction

One of the most popular sports in the United States of America is football, commonly referred to as the National Football League (NFL). Millions of fans tune in every Sunday to watch their favorite sport, favorite team, and favorite players. Throughout the NFL, there are many statistics that help coaches, scouts, and fans analyze how well their respective teams are doing, such as the average amount of points scored, how many rushing yards a team allows, or how many passing yards a team has per game. This tutorial will explore the statistics and data of each team during the 2019 NFL season to help make conclusions on how different variables, such as passing yards, can impact each team differently. The data from the 2019 season will come from the “ProFootballReference” website: https://www.pro-football-reference.com/years/2019/index.htm

!conda install -y pandas
!conda install -y seaborn
!conda install -y scikit-learn
!conda install -y statsmodels
!conda install -y seaborn
!conda install -y nbconvert
import pandas as pd
from sklearn import linear_model
import statsmodels.api
import seaborn as sns 
import glob
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import paired_ttest_kfold_cv
from sklearn.utils import shuffle
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score



