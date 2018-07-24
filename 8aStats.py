import sqlite3
import pandas as pd
import numpy as np
from numpy import array
from scipy.stats import norm
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

conn = sqlite3.connect("C:\8a.db")

#print the tables
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())

db = sqlite3.connect("C:\8a.db")
table = pd.read_sql_query("SELECT grade_id from ascent WHERE climb_type = 0", db)
df2 = pd.DataFrame(table)


df3 = df2["grade_id"]


data = df3
print data

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=50, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
print "xmin = ", xmin
print "xmax = ", xmax
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.xticks(np.arange(min(x), max(x)+1, 5.0))
title = "Route Ascent Distribution | Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()
