import os
import time

import matplotlib.pyplot as plt
import pandas as pd




class BoxPlot:
    def __init__(self, PLOTS, name, ax, labels):
        self.ax = ax
        self.labels = labels
        PLOTS[name] = self

    def update(self, box_plot_data):
        self.ax.cla()
        self.ax.tick_params(axis='y', labelsize=6)
        self.ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        sample_sizes = [len(s) for s in box_plot_data ]
        widths = [0.9 * s / max(sample_sizes) for s in sample_sizes]
        self.ax.boxplot(box_plot_data, whis=2, patch_artist=True ,labels=self.labels, widths = widths, flierprops={'marker': '+'})


class APlot():

    def __init__(self, PLOTS, name, ax, line_def, legend =True):
        self.ax = ax
        PLOTS[name] = self
        self.lines =[]
        for l in line_def:
            line, = ax.plot ([], [], **l)
            self.lines.append (line)
        if legend: ax.legend ()
        self.ax.tick_params (axis='y', labelsize=6)

    def update(self, l_idx, x, y):
        line = self.lines[l_idx]
        line.set_data (x, y)

    def getLims(self):
        xlim = (1e6, -1e6)
        ylim = (1e6, -1e6)
        for line in self.lines:
            xx, yy = line.get_data ()
            xlim = (min (xx[0], xlim[0]), max (xx[-1], xlim[1]))
            ylim = (min (min (yy), ylim[0]), max (max (yy), ylim[1]))

        xpad = (xlim[1] - xlim[0]) * 0.05
        ypad = (ylim[1] - ylim[0]) * 0.1
        xlim = (xlim[0] - xpad, xlim[1] + xpad)
        ylim = (ylim[0] - ypad, ylim[1] + ypad)
        return xlim, ylim

    def refresh(self):

        x, y = self.getLims ()
        self.ax.set_xlim (x[0], x[1])
        self.ax.set_ylim (y[0], y[1])


class EpisodePlot:

    def __init__(self, title="Episode", plot_dir = None):
        self.save_file = None
        if plot_dir:
            os.makedirs (plot_dir, exist_ok=True)
            self.save_file =os.path.join(plot_dir, "train{}.png" )

        self.PLOTS = {}
        self.rwd = []
        self.asset = []
        self.risk = []
        plt.ion ()
        self.fig = plt.figure (figsize=(7, 5), constrained_layout=True)
        self.fig.canvas.set_window_title(title)
        self.init_fig ()

    def init_fig(self):
        grid_ax = self.fig.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1],
                                         'height_ratios': [3, 1]})

        lines = [
            {'color': 'g', 'label': 'Account', 'alpha': 0.8},
            {'color': 'r', 'label': 'Unrealized', 'alpha': 0.8},
            {'color': 'b', 'label': 'Total', 'alpha': 0.6},
            {'color': 'gray', 'label': 'Reward', 'alpha': 0.6}]
        APlot (self.PLOTS, "all", grid_ax[0,0], lines)

        BoxPlot (self.PLOTS, "posneg", grid_ax[0,1], ['Pos', 'Neg'])

        lines = [{'label': 'Position', 'lw': 0.5}]
        pp = APlot (self.PLOTS, "pos", grid_ax[1,0], lines, legend=False)
        lines = [{'color': 'r', 'label': 'Unit', 'lw': 0.5}]
        APlot (self.PLOTS, "unit", pp.ax.twinx (), lines)

        empty = grid_ax[1,1]
        empty.set_xticklabels ([])
        empty.set_yticklabels ([])


    def _update_line(self, name, l_idx, x, y):
        if (isinstance (y, pd.Series)): y = y.tolist ()
        x = range (1, len (x) + 1)

        self.PLOTS[name].update (l_idx, x, y)

    def _update_box(self, name, data):
        self.PLOTS[name].update (data)

    def refresh(self):
        for key, ax in self.PLOTS.items ():
            if isinstance (ax, APlot): ax.refresh ()

        self.fig.canvas.draw ()
        self.fig.canvas.flush_events ()
        time.sleep (0.1)

    def update(self, iteration, **info):
        idx = info['idx']
        pos = info['pos']
        neg = info['neg']

        self.fig.suptitle ('Train iteration {}'.format (iteration), fontsize=11)
        self._update_line ("all", 0, idx, info["cash"])
        self._update_line ("all", 1, idx, info["unreal"])
        self._update_line ("all", 2, idx, info["asset"])
        self._update_line ("all", 3, idx, info["reward"])

        self._update_line ('pos', 0, idx, info["position"])
        self._update_line ("unit", 0, idx, info["unit"])
        self._update_box ('posneg', [pos, neg])
        self.refresh ()
        if self.save_file:
            self.fig.savefig(self.save_file.format(iteration))
            print("Plot Saved: ", self.save_file.format(iteration))

