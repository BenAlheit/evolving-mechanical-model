"""
Author: Benjamin Alheit
Email: alhben001@myuct.ac.za

Plots the dissipation of a single Maxwell latch while latch is released (Figure 14)

"""

import numpy as np
from plotting_config import *

format_matplotlib()
x = np.linspace(0, 4, 10000)
plt.plot(x, 1-np.exp(-2*x), color=CMAP(0.1), label='$t_r=0$, $\\tau_i=1$')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$\dfrac{2D}{E \varepsilon_{sl}^2}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./figures/dissipation.pdf')
plt.show()
