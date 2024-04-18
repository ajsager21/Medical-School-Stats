#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install pandas numpy statsmodels


# In[28]:


import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[29]:


df = pd.read_csv('Med School.csv')


# In[30]:


# Independent variables
X = df[['nMCAT', 'nGPA', 'nClinical Hours', 'nResearch Hours', 'nStudent-Athlete?', 'nShadowing Hours']]

# Dependent variables
Y_secondary = df['Secondary %']
Y_interview = df['Interview %']
Y_waitlist = df['Waitlist %']
Y_acceptance = df['Acceptance %']


# In[31]:


df.dropna(inplace=True)  # This will remove rows with any missing values


# In[32]:


# Drop rows with missing values
df.dropna(inplace=True)

# Check if there's still data in the DataFrame
if df.empty:
    print("DataFrame is empty after removing missing values.")
else:
    # Re-define independent and dependent variables
    X = df[['MCAT', 'Clinical Hours','Research Hours', 'GPA', 'Student-Athlete?', 'Shadowing Hours']]
    Y_secondary = df['Secondary %']
    Y_interview = df['Interview %']
    Y_waitlist = df['Waitlist %']
    Y_acceptance = df['Acceptance %']

    # Run regressions for each dependent variable
    run_regression(X, Y_secondary, 'Secondary %')
    run_regression(X, Y_interview, 'Interview %')
    run_regression(X, Y_waitlist, 'Waitlist %')
    run_regression(X, Y_acceptance, 'Acceptance %')



# In[33]:


# Define a function to run the regression and print coefficients
def print_regression_equation(X, Y, dependent_variable_name):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    coefficients = model.params[1:]  # Exclude intercept
    intercept = model.params[0]
    
    print(f"Regression equation for {dependent_variable_name}:")
    equation = f"{dependent_variable_name} = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + ({coef:.4f}) * {X.columns[i]}"
    print(equation)

# Print regression equations for each dependent variable
print_regression_equation(X, Y_secondary, 'Secondary %')
print_regression_equation(X, Y_interview, 'Interview %')
print_regression_equation(X, Y_waitlist, 'Waitlist %')
print_regression_equation(X, Y_acceptance, 'Acceptance %')


# In[34]:


#How impactful is student-athlete status?


# In[36]:


# Define a function to run the regression and print coefficients
def print_regression_results(X, Y, dependent_variable_name):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    coef_student_athlete = model.params['nStudent-Athlete?']
    p_value_student_athlete = model.pvalues['nStudent-Athlete?']
    
    print(f"Regression results for {dependent_variable_name}:")
    print(f"Coefficient for 'Student-Athlete?': {coef_student_athlete:.4f}")
    print(f"P-value for 'Student-Athlete?': {p_value_student_athlete:.4f}")
    print("")

# Print regression results for each dependent variable
print_regression_results(X_with_student_athlete, Y_secondary, 'Secondary %')
print_regression_results(X_with_student_athlete, Y_interview, 'Interview %')
print_regression_results(X_with_student_athlete, Y_waitlist, 'Waitlist %')
print_regression_results(X_with_student_athlete, Y_acceptance, 'Acceptance %')



# In[ ]:




