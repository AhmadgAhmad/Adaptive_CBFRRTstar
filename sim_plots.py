import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib._color_data as mcd
import numpy as np
from agent import Agent
from obstacle import Obstacle
from params import *

class Cbf_data:

    def __init__(self,agents,obsts):
        self.params = Params()
        self.agents = agents
        self.obsts = obsts
        self.a2a = list() # stores agent to agent cbf values
        self.a2o = list() # stores agent to obstacle values
        self.a2a_constr = list() # Stores the constraint values for the agent/agent cbfs
        self.a2o_constr = list() # Stores the constraint values for the agent/obst cbfs

        self.cm = ColorManager()
        # if len(agents) > 1:
        #     self.aa_colors = cm.get_colors(0,len(agents)**2)
        #     self.ao_colors = cm.get_colors(len(agents), len(agents)*len(obsts))
        # else:
        #     self.aa_colors = None
        #     self.ao_colors = cm.get_colors(0, len(agents)*len(obsts))

        if self.params.live_plots:
            self.make_axes()
        else:
            self.cbf_ax = None
            self.constr_ax = None

        # Initialize arrays to save cbf data
        for i in range(len(agents)):
            aa = list()
            aac = list()
            for j in range(len(agents)):
                aa.append(list())
                aac.append(list())

            ao = list()
            aoc = list()
            for k in range(len(obsts)):
                ao.append(list())
                aoc.append(list())
            
            self.a2a.append(aa)
            self.a2a_constr.append(aac)
            self.a2o.append(ao)
            self.a2o_constr.append(aoc)

    def make_axes(self):
        if self.params.plot_cbf and self.params.plot_constrs:
            _,axs = plt.subplots(2,1)
            self.cbf_ax = axs[0]
            self.constr_ax = axs[1]
        elif self.params.plot_clf:
            _,self.cbf_ax = plt.subplots()
        elif self.params.plot_delta:
            _,self.constr_ax = plt.subplots()

    def add_cbf_val(self,val,agent,obst):
        if type(obst) is Agent:
            i = min([agent.id,obst.id])
            j = max([agent.id,obst.id])
            self.a2a[i][j].append(val)
        else:
            self.a2o[agent.id][obst.id].append(val)

    def add_constr_val(self,val,agent,obst):
        if type(obst) is Agent:
            i = min([agent.id,obst.id])
            j = max([agent.id,obst.id])
            self.a2a_constr[i][j].append(val)
        else:
            self.a2o_constr[agent.id][obst.id].append(val)

    def plot(self,time,save=False):
        if self.params.plot_cbf:
            if self.cbf_ax is None:
                self.make_axes()
            self.plot_cbf(time,save)
        if self.params.plot_constrs:
            if self.constr_ax is None:
                self.make_axes()
            self.plot_constrs(time,save)

    def plot_cbf(self, time, save=False):
        labels = list()
        if self.cbf_ax is None:
            _, self.cbf_ax = plt.subplots()
        for i in range(len(self.a2a)):
            for j in range(len(self.a2a[i])):
                vals = self.a2a[i][j]
                if len(vals) != 0:
                    self.cbf_ax.plot(np.array(time), np.array(np.array(vals)), color=self.cm.get_colors())
                    labels.append('Agent{} Agent{}'.format(i,j))

        for i in range(len(self.a2o)):
            for k in range(len(self.a2o[i])):
                vals = self.a2o[i][k]
                if len(vals) != 0:
                    self.cbf_ax.plot(np.array(time), np.array(np.array(vals)), color=self.cm.get_colors())
                    labels.append('Agent{} Obst{}'.format(i,j))
        
        # self.cbf_ax.legend(labels)
        if save:
            with PdfPages('z_cbf_plot.pdf') as pdf:
                pdf.savefig()
        else:
            self.cbf_ax.set_title("Control Barrier Function Values")

    def plot_constrs(self, time, save=False):
        labels = list()
        if self.constr_ax is None:
            _, self.constr_ax = plt.subplots()
        for i in range(len(self.a2a_constr)):
            for j in range(len(self.a2a_constr[i])):
                vals = self.a2a_constr[i][j]
                if len(vals) != 0:
                    self.constr_ax.plot(np.array(time), np.array(np.array(vals)), color=self.aa_colors[i,j])
                    labels.append('Agent{} Agent{}'.format(i,j))

        for i in range(len(self.a2o_constr)):
            for k in range(len(self.a2o_constr[i])):
                vals = self.a2o_constr[i][k]
                if len(vals) != 0:
                    self.constr_ax.plot(np.array(time), np.array(np.array(vals)), color=self.ao_colors[i,k])
                    labels.append('Agent{} Obst{}'.format(i,j))
        
        self.constr_ax.legend(labels)
        if save:
            with PdfPages('z_constr_plot.pdf') as pdf:
                pdf.savefig()
        else:
            self.constr_ax.set_title("CBF Constraint Values")

class Clf_data:
    def __init__(self,agents):
        self.params = Params()
        self.clf_data = list()
        self.delta_data = list()
        cm = ColorManager()
        self.colors = cm.get_colors(0,len(agents))
        
        if self.params.live_plots:
            self.make_axes()
        else:
            self.clf_ax = None
            self.delta_ax = None

        for i in range(len(agents)):
            if agents[i].goal is None:
                self.clf_data.append(None)
                self.delta_data.append(None)
            else:
                self.clf_data.append([])
                self.delta_data.append([])

    def make_axes(self):
        if self.params.plot_clf:
            self.clf_fig,self.clf_ax = plt.subplots()
        if self.params.plot_delta:
            self.delta_fig,self.delta_ax = plt.subplots()

    def add_clf_val(self,val,idx):
        self.clf_data[idx].append(val)

    def add_delta_val(self,val,idx):
        self.delta_data[idx].append(val)

    def plot(self,time,save=False):
        if self.params.plot_clf:
            if self.clf_ax is None:
                self.make_axes()
            self.plot_clf(time,save)
        if self.params.plot_delta:
            if self.delta_ax is None:
                self.make_axes()
            self.plot_delta(time,save)

    def plot_clf(self,time,save=False):
        labels = list()
        if self.clf_ax is None:
            _, self.clf_ax = plt.subplots()
        else:
            self.clf_ax.cla()

        for i in range(len(self.clf_data)):
            vals = self.clf_data[i]
            if len(vals) != 0:
                self.clf_ax.plot(np.array(time), np.array(np.array(vals)), color=self.colors[i])
                labels.append('Agent{}'.format(i))

        self.clf_ax.legend(labels)
        if save:
            with PdfPages('z_clf_plot.pdf') as pdf:
                pdf.savefig(self.clf_fig)
        else:
            self.clf_ax.set_title("Control Lyapunov Function Values")

        

    def plot_delta(self,time,save=False):
        labels = list()
        if self.delta_ax is None:
            _, self.delta_ax = plt.subplots()
        else:
            self.delta_ax.cla()

        for i in range(len(self.delta_data)):
            vals = self.delta_data[i]
            if len(vals) != 0:
                self.delta_ax.plot(np.array(time), np.array(np.array(vals)), color=self.colors[i])
                labels.append('Agent{}'.format(i))

        self.delta_ax.legend(labels)
        if save:
            with PdfPages('z_delta_plot.pdf') as pdf:
                pdf.savefig()
        else:
            self.delta_ax.set_title("CLF Relaxation Variable Values")

class ColorManager(object):
    def __init__(self):
        colors = list()
        colors.append(mcd.XKCD_COLORS["xkcd:blue"])
        colors.append(mcd.XKCD_COLORS["xkcd:dark orange"])
        colors.append(mcd.XKCD_COLORS["xkcd:jungle green"])
        colors.append(mcd.XKCD_COLORS["xkcd:ochre"])
        colors.append(mcd.XKCD_COLORS["xkcd:aquamarine"])
        colors.append(mcd.XKCD_COLORS["xkcd:dark pink"])
        colors.append(mcd.XKCD_COLORS["xkcd:barney purple"])
        colors.append(mcd.XKCD_COLORS["xkcd:rusty red"])
        colors.append(mcd.XKCD_COLORS["xkcd:sea blue"])
        colors.append(mcd.XKCD_COLORS["xkcd:red brown"])
        colors.append(mcd.XKCD_COLORS["xkcd:blue green"])
        colors.append(mcd.XKCD_COLORS["xkcd:green yellow"])
        colors.append(mcd.XKCD_COLORS["xkcd:greyish green"])
        colors.append(mcd.XKCD_COLORS["xkcd:pumpkin"])
        colors.append(mcd.XKCD_COLORS["xkcd:dusty purple"])
        colors.append(mcd.XKCD_COLORS["xkcd:steel grey"])
        colors.append(mcd.XKCD_COLORS["xkcd:sepia"])
        colors.append(mcd.XKCD_COLORS["xkcd:purplish grey"])
        self.colors = colors
        self.idx = 0
    
    def get_colors(self, idx=None, num=None):
        if idx is None:
            col = self.colors[self.idx]
            self.idx += 1
            if self.idx == len(self.colors):
                self.idx = 0
            return col
        elif num is None:
            return self.colors[idx]
        else:
            return self.colors[idx:idx+num]

if __name__ == '__main__':
    cm = ColorManager()
    plt.plot((0,10),(1,1))#, color=cm.get_color(0))
    plt.plot((0,10),(0,2))#, color=cm.get_color(1))
    plt.show()
