import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

daily = pd.read_csv('4.DTW_prec.csv',header = 'infer')
daily.index = pd.to_datetime(daily['DATE'])
daily = daily['PRCP']
ax = daily.plot(kind='line',figsize=(15,3))
print(ax.set_title('Daily Precipitation (variance = %.4f)' % (daily.var())))
plt.show()