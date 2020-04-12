import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

daily = pd.read_csv('4.DTW_prec.csv',header = 'infer')
daily.index = pd.to_datetime(daily['DATE'])
daily = daily['PRCP']
annual = daily.groupby(pd.Grouper(freq='Y')).sum()
ax = annual.plot(kind='line',figsize=(15,3))
ax.set_title('Annual Precipitation (variance = %.4f)' % (annual.var()))
plt.show()