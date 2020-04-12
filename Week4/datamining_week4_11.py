import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

daily = pd.read_csv('4.DTW_prec.csv',header = 'infer')
daily.index = pd.to_datetime(daily['DATE'])
daily = daily['PRCP']
monthly = daily.groupby(pd.Grouper(freq='M')).sum()
ax = monthly.plot(kind='line',figsize=(15,3))
ax.set_title('Monthly Precipitation (variance = %.4f)' % (monthly.var()))
plt.show()