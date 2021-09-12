"""
Author: Benjamin Alheit
Email: alhben001@myuct.ac.za

Code to display the mechanical behaviour of a 1D compound spring device that evolves with loading history.

"""

import numpy as np
from plotting_config import *
import matplotlib.lines as mlines

SAVE_FIGS = True
FORMAT = True


class PreloadedSpring:
    def __init__(self, eps_pf_i: float, E_pf_i: float, tau_i: float, Es_i: float):
        """
        A preloaded spring object.
        :param eps_pf_i: Initial strain at which preloaded spring is released.
        :param E_pf_i: Stiffness of preloaded spring.
        :param tau_i: Relaxation time of maxwell element controlling evolution of latch.
        :param Es_i: Stiffness of spring in maxwell element.
        """
        self.E_pf_i = E_pf_i
        self.tau_i = tau_i
        self.Es_i = Es_i

        self.eps_s = eps_pf_i
        self.eps_sl = eps_pf_i

        self.t_release = None
        self.released = False

        self.D_current = 0
        self.D_last_release = 0

    def get_E_pf_i(self, eps: float, t: float) -> float:
        """
        Determines whether spring is attached to base spring or not and calculates dissipation of Maxwell element
        :param eps: Current strain of overall compound device
        :param t: Time
        :return: Stiffness of preloaded spring if attached to base spring, otherwise 0
        """
        if not self.released and eps >= self.eps_s:
            self.released = True
            self.t_release = t

        if self.released:
            self.eps_s = self.eps_sl * np.exp(-(t - self.t_release) / self.tau_i)
            self.D_current = self.D_last_release + 0.5 * self.Es_i * self.eps_sl ** 2 * (
                    1. - np.exp(-2 * (t - self.t_release) / self.tau_i))

            if eps < self.eps_s:
                self.released = False
                self.eps_sl = self.eps_s
                self.D_last_release = self.D_current

        if self.released:
            return self.E_pf_i
        else:
            return 0


class CompoundEvolvingSpring:
    def __init__(self, Eb: float, preloaded_springs: [PreloadedSpring]):
        """
        Compound spring with evolving preloaded springs.
        :param Eb: The stiffness of the base spring
        :param preloaded_springs: A list of preloaded springs in the compound spring
        """
        self.Eb = Eb
        self.preloaded_springs = preloaded_springs

    def increment_model(self, eps: float, t: float) -> (float, float):
        """
        Updates the state of the preloaded springs for a give strain and time, and calculates the current stress and
        dissipation of the device.
        :param eps: Strain
        :param t: Time
        :return: Stress and dissipation
        """
        Eeff = self.Eb + np.sum(
            list(map(lambda preloaded_spring: preloaded_spring.get_E_pf_i(eps, t), self.preloaded_springs)))
        D = np.sum(list(map(lambda preloaded_spring: preloaded_spring.D_current, self.preloaded_springs)))
        return Eeff * eps, D

    def run_load(self, strain_path: np.array, t_array: np.array, return_dissipation: bool = False):
        """
        Calculates the stress (and optionally) the dissipation for a given loading path.
        :param strain_path: The strain during the loading path.
        :param t_array: The time corresponding to the strain values.
        :param return_dissipation: Whether or not to return the dissipation values
        :return: stress (and optionally, dissipation)
        """
        stress_and_disp = np.array([self.increment_model(eps, t) for (eps, t) in zip(strain_path, t_array)])
        if return_dissipation:
            return stress_and_disp[:, 0], stress_and_disp[:, 1]
        else:
            return stress_and_disp[:, 0]


def get_loading_path(
        eps_f: float = 1,
        n_incs_per_half_cycle: int = 1000,
        n_cycles: int = 5,
        t_cycle: float = 4) -> (np.array, np.array):
    """
    Constructs a cyclical loading path.
    :param eps_f: The maximum strain in the loading path.
    :param n_incs_per_half_cycle: Number of increments per half cycle
    :param n_cycles: Number of cycles
    :param t_cycle: The time taken for one cycle
    :return: The strains and times for the loading path
    """
    t = np.linspace(0, t_cycle * n_cycles, n_incs_per_half_cycle * 2 * n_cycles)
    strain_half_cycle = np.linspace(0, eps_f, n_incs_per_half_cycle)
    strain_cycle = np.concatenate((strain_half_cycle, strain_half_cycle[::-1]))
    strains = np.concatenate([strain_cycle for i in range(n_cycles)])
    return strains, t


def construct_compound_spring(E_b: float = 0.1,
                              E_pf: float = 1,
                              eps_pf: float = 1,
                              tau: float = 3,
                              E_s: float = 1,
                              a: float = 0.1,
                              n_preloaded: int = 1000) -> CompoundEvolvingSpring:
    """
    Constructs a compound spring object with given preloaded springs.
    :param E_b: Stiffness of the base spring.
    :param E_pf: Cumulative stiffness of the preloaded springs.
    :param eps_pf: Final strain of the preloaded springs.
    :param eps_f: Final strain of the loading cycle.
    :param tau: Relaxation time of evolving Maxwell latches.
    :param E_s: Stiffness of springs in evolving Maxwell latches
    :param a: Parameter for distribution of preloaded springs.
    :param n_preloaded: Number of preloaded springs.
    :return:
    """
    return CompoundEvolvingSpring(E_b,
                                  [PreloadedSpring((i + 1) ** a * eps_pf / n_preloaded ** a,
                                                   E_pf / n_preloaded, tau, E_s / n_preloaded)
                                   for i in range(n_preloaded)])


def plot(x: np.array,
         y: np.array,
         n_cycles: int,
         x_label: str = '',
         y_label: str = '',
         title: str = '',
         save: bool = False,
         fig_name: str = None
         ):
    """
    Plots data resulting from cyclical loading
    :param x: x data to plot
    :param y: y data to plot
    :param n_cycles: Number of loading cycles for data
    :param x_label: self-evident
    :param y_label: self-evident
    :param title: self-evident
    :param save: Whether to save the figure
    :param fig_name: Whether to save the figure
    :return: None
    """

    ys = np.array_split(y, n_cycles)
    xs = np.array_split(x, n_cycles)
    plt.figure()
    colors = [CMAP(i / n_cycles) for i in range(n_cycles)]
    legend_items = [
        mlines.Line2D([], [], color='black', linewidth=3, linestyle='--', label='Loading'),
        mlines.Line2D([], [], color='black', linewidth=1, linestyle='-', label='Unloading')]
    for i in range(n_cycles):
        xs_loading, xs_unloading = np.array_split(xs[i], 2)
        ys_loading, ys_unloading = np.array_split(ys[i], 2)
        plt.plot(xs_loading, ys_loading, color=colors[i], linewidth=3, linestyle='--')
        plt.plot(xs_unloading, ys_unloading, color=colors[i])
        legend_items.append(
            mlines.Line2D([], [], color=colors[i], linewidth=0, marker='s', markersize=15, label=f'Cycle {i + 1}'))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend(handles=legend_items)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'figures/{fig_name}.pdf')


def plot_stress_curve(
        E_b: float = 0.1,
        E_pf: float = 1,
        eps_pf: float = 1,
        eps_f: float = 1,
        tau: float = 3,
        E_s: float = 1,
        a: float = 0.1,
        n_preloaded: int = 1000,
        n_incs_per_half_cycle: int = 1000,
        n_cycles: int = 5,
        t_cycle: float = 4,
        title: str = '',
        save: bool = False,
        fig_name: str = None):
    """
    Plots the stress-strain curve of a compound spring for a given loading cycle.

    :param E_b: Stiffness of the base spring.
    :param E_pf: Cumulative stiffness of the preloaded springs.
    :param eps_pf: Final strain of the preloaded springs.
    :param eps_f: Final strain of the loading cycle.
    :param tau: Relaxation time of evolving Maxwell latches.
    :param E_s: Stiffness of springs in evolving Maxwell latches
    :param a: Parameter for distribution of preloaded springs.
    :param n_preloaded: Number of preloaded springs.
    :param n_incs_per_half_cycle: Number of increments in half a loading cycle.
    :param n_cycles: Number of loading cycles.
    :param t_cycle: Time taken for each loading cycle.
    :param title: Title of the plotted figure.
    :param save: Whether or not to save the figure.
    :param fig_name: If saving the figure, the name to save the figure as.
    :return: None
    """
    strains, t = get_loading_path(eps_f, n_incs_per_half_cycle, n_cycles, t_cycle)
    compound_spring = construct_compound_spring(E_b, E_pf, eps_pf, tau, E_s, a, n_preloaded)
    stress = compound_spring.run_load(strains, t)
    plot(strains, stress, n_cycles, r'$\varepsilon$', r'$\sigma$ (Pa)', title=title, save=save, fig_name=fig_name)


def plot_disp_curve(
        E_b: float = 0.1,
        E_pf: float = 1,
        eps_pf: float = 1,
        eps_f: float = 1,
        tau: float = 3,
        E_s: float = 1,
        a: float = 0.1,
        n_preloaded: int = 1000,
        n_incs_per_half_cycle: int = 1000,
        n_cycles: int = 5,
        t_cycle: float = 4,
        title: str = '',
        save: bool = False,
        fig_name: str = None):
    """
    Plots the dissipation-time curve of a compound spring for a given loading cycle.

    :param E_b: Stiffness of the base spring.
    :param E_pf: Cumulative stiffness of the preloaded springs.
    :param eps_pf: Final strain of the preloaded springs.
    :param eps_f: Final strain of the loading cycle.
    :param tau: Relaxation time of evolving Maxwell latches.
    :param E_s: Stiffness of springs in evolving Maxwell latches
    :param a: Parameter for distribution of preloaded springs.
    :param n_preloaded: Number of preloaded springs.
    :param n_incs_per_half_cycle: Number of increments in half a loading cycle.
    :param n_cycles: Number of loading cycles.
    :param t_cycle: Time taken for each loading cycle.
    :param title: Title of the plotted figure.
    :param save: Whether or not to save the figure.
    :param fig_name: If saving the figure, the name to save the figure as.
    :return: None
    """
    strains, t = get_loading_path(eps_f, n_incs_per_half_cycle, n_cycles, t_cycle)
    compound_spring = construct_compound_spring(E_b, E_pf, eps_pf, tau, E_s, a, n_preloaded)
    stress, disp = compound_spring.run_load(strains, t, return_dissipation=True)
    plot(t, disp, n_cycles, r'$t$ (s)', r'$D$ (J)', title=title, save=save, fig_name=fig_name)


def main():
    if FORMAT:
        format_matplotlib()

    # Plot single compound spring (Figure 5)
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=1, n_cycles=1, save=SAVE_FIGS, fig_name='single-spring-1-cycle')
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=1, n_cycles=2, save=SAVE_FIGS, fig_name='single-spring-2-cycles')

    # Plot multicompound spring (Figure 7)
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=1, n_cycles=1, save=SAVE_FIGS, fig_name='spring-1')
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=2, n_cycles=1, save=SAVE_FIGS, fig_name='spring-2')
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=10, n_cycles=1, save=SAVE_FIGS, fig_name='spring-10')
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=1000, n_cycles=1, save=SAVE_FIGS, fig_name='spring-1000')

    # Display a effect with varying n (Figure 8)
    plot_stress_curve(tau=0.00001, a=0.2, n_preloaded=2, n_cycles=1, save=SAVE_FIGS, fig_name='a-02-n-2')
    plot_stress_curve(tau=0.00001, a=0.2, n_preloaded=5, n_cycles=1, save=SAVE_FIGS, fig_name='a-02-n-5')
    plot_stress_curve(tau=0.00001, a=0.2, n_preloaded=10, n_cycles=1, save=SAVE_FIGS, fig_name='a-02-n-10')
    plot_stress_curve(tau=0.00001, a=0.2, n_preloaded=1000, n_cycles=1, save=SAVE_FIGS, fig_name='a-02-n-1000')

    # Plot varying a (Figure 9)
    plot_stress_curve(tau=0.00001, a=1, n_preloaded=1000, n_cycles=1, save=SAVE_FIGS, fig_name='a-1-n-1000')
    plot_stress_curve(tau=0.00001, a=0.5, n_preloaded=1000, n_cycles=1, save=SAVE_FIGS, fig_name='a-05-n-1000')
    plot_stress_curve(tau=0.00001, a=0.1, n_preloaded=1000, n_cycles=1, save=SAVE_FIGS, fig_name='a-01-n-1000')
    plot_stress_curve(tau=0.00001, a=0.05, n_preloaded=1000, n_cycles=1, save=SAVE_FIGS, fig_name='a-005-n-1000')

    # Plot evolving latches with varying n (Figure 12)
    plot_stress_curve(a=0.1, n_preloaded=1, n_cycles=2, save=SAVE_FIGS, fig_name='temp-evolution-1')
    plot_stress_curve(a=0.1, n_preloaded=2, n_cycles=2, save=SAVE_FIGS, fig_name='temp-evolution-2')
    plot_stress_curve(a=0.1, n_preloaded=10, n_cycles=2, save=SAVE_FIGS, fig_name='temp-evolution-10')
    plot_stress_curve(a=0.1, n_preloaded=1000, n_cycles=2, save=SAVE_FIGS, fig_name='temp-evolution-1000')

    # Plot evolving latches with varying n=1000 fir 7 cycles (Figure 13)
    plot_stress_curve(a=0.1, n_preloaded=1000, n_cycles=7, save=SAVE_FIGS, fig_name='temp-evolution-1000-cycles')

    # Plot dissipation for Figure 12 (Figure 15)
    plot_disp_curve(a=0.1, n_preloaded=1, n_cycles=2, save=SAVE_FIGS, fig_name='disp-1')
    plot_disp_curve(a=0.1, n_preloaded=2, n_cycles=2, save=SAVE_FIGS, fig_name='disp-2')
    plot_disp_curve(a=0.1, n_preloaded=10, n_cycles=2, save=SAVE_FIGS, fig_name='disp-10')
    plot_disp_curve(a=0.1, n_preloaded=1000, n_cycles=2, save=SAVE_FIGS, fig_name='disp-1000')

    # Plot dissipation for Figure 13 (Figure 16)
    plot_disp_curve(a=0.1, n_preloaded=1000, n_cycles=7, save=SAVE_FIGS, fig_name='disp-1000-cycles')

    plt.show()


if __name__ == '__main__':
    main()
