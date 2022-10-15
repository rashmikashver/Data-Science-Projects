
#______________________________________________________
# Q1.
import pandas as pd
data = pd.read_csv('landslide_data3.csv')
data = data.set_index(['stationid'])
del data['dates']

print('1. Mean :')
print(data.mean(),'\n')
print('Median :')
print(data.median(),'\n')
print('Mode :')
print(data.mode(),'\n')
print('Minimum :')
print(data.min(),'\n')
print('Maximum :')
print(data.max(),'\n')
print('Standard Deviation :')
print(data.std(),'\n')

# Q2.(a)_________________________________
import matplotlib.pyplot as plt

print('2.(a)')
attributes1 = ['temperature','humidity','pressure','lightavgw/o0','lightmax','moisture']
for i in attributes1:
    data.plot.scatter(x='rain',y=i,alpha=0.9)
    plt.show()     # It will show scatter plot b/w 'rain' & other attributes

# (b)    
print('2.(b)')
attributes2 = ['rain','humidity','pressure','lightavgw/o0','lightmax','moisture']
for i in attributes2:
    data.plot.scatter(x='temperature',y=i,alpha=0.9)
    plt.show()    # It will show scatter plot b/w 'temperature' & other attributes

#____________________________________________________
# Q3. (a)
corr = data.corr(method='pearson')      # gives Pearson corr. coeff. b/w all attributes
print("\n3.(a) Correlation Coefficient of 'rain' with:\n")
print(corr['rain'])
# (b)
print("\n3.(b) Correlation Coefficient of 'temperature' with:\n")
print(corr['temperature'])
    
#___________________________________________________
# Q4.
# Histogram for the attributes ‘rain’ and ‘moisture'
print('\n4. Histogram for the attributes ‘rain’ and ‘moisture’:')

data.hist(column='rain',bins=10)
plt.xlabel('Measure of rainfall in ml')
plt.ylabel('frequency')
plt.show()

data.hist(column='moisture',bins=10)
plt.xlabel('% of water stored in soil')
plt.ylabel('frequency')
plt.show()   

#____________________________________________________
# Q5.
# Histogram of attribute ‘rain’ for each of the 10 stations 
print('\n5. Histogram of ‘rain’ for each of the 10 stations (t10, t11, t12, t13, t14, t15, t6, t7, t8, t9):')
data.hist(column='rain',by='stationid',layout=(5,2),figsize=(12,18))
plt.xlabel('rain(ml)')
plt.ylabel('frequency')
plt.show()

#___________________________________________________    
# Q6.   
# Boxplot of 'rain' 
print("\n6. Boxplot of 'rain':" )

data.boxplot(column='rain')                    # with Outliers
plt.ylabel('Measure of rainfall in ml')
plt.title("Boxplot of 'rain' (with Outliers)")
plt.show()
data.boxplot(column='rain',showfliers=False)   # Outliers are not shown
plt.ylabel('Measure of rainfall in ml')
plt.title("Boxplot of 'rain' (Hidden Outliers)")
plt.show()

print("\nBoxplot of 'moisture': ")
data.boxplot(column='moisture')     # Boxplot of 'moisture' 
plt.ylabel('% of water stored in soil')
plt.title("Boxplot of 'moisture'")
plt.show()

#_______________________________________________________    
    
    
    
    
    
