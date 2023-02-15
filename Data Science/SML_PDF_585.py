# https://whoishoo.tistory.com/585
# Library
import numpy as np
import pandas as pd
from scipy.stats import norm, t
import matplotlib.pyplot as plt

#μX=μY= 0,σX=σY= 1
muX=0
muY=0
sigmaX=1
sigmaY=1

#interval [-3,3]
xGrid=np.arange(-3,3,.01)
yGrid=np.arange(-3,3,.01)

#Normal X Pdf
plt.figure()
plt.plot(xGrid, norm.pdf(xGrid,muX,sigmaX))
plt.title("Marginal PDF(X)")
plt.show()

# Normal Y Pdf
plt.figure()
plt.plot(yGrid, norm.pdf(yGrid, muY,sigmaY))
plt.title("Marginal PDF(Y)")
plt.show()

pGrid=np.arange(-0.75,1,0.25)
range=np.arange(-10,10,0.05)

# we already know the μX=μY= 0,σX=σY= 1 values(assignment2.1 code)

plt.figure()
for p in pGrid:
  mu = muX+p*sigmaX*(1-muY)/sigmaY
  sigma = (1-p**2)*sigmaX**2
  plt.plot(range,norm.pdf(range,mu,sigma), label=p)
plt.legend()
plt.show()
