"""
Author: Benjamin Alheit
Email: alhben001@myuct.ac.za

Plots the evolution of a single Maxwell latch while latch is released (Figure 11)

"""
import numpy as np
from plotting_config import *

format_matplotlib()
x = np.linspace(0, 6, 10000)
plt.plot(x, np.exp(-x), color=CMAP(0.1), label='$t_r=0$, $\\tau_i=1$')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$\dfrac{\varepsilon_s}{\varepsilon_{sl}}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./figures/es.pdf')
plt.show()
