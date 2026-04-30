#!/usr/bin/env python3
"""Advanced electrical engineering GUI for propulsion motor analysis and simulation."""
import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MotorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("14 MW Propulsion Motor - Advanced Engineering Lab")
        self.geometry("1500x920")
        self.minsize(1100, 700)

        self.running = False
        self.sim_t = np.array([])
        self.sim_w = np.array([])
        self.sim_i = np.array([])
        self.after_id = None

        self._build_ui()
        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Advanced Electrical Engineering Workbench", font=("Segoe UI", 14, "bold")).pack(side="left")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_main = ttk.Frame(self.nb)
        self.tab_model = ttk.Frame(self.nb)
        self.tab_fault = ttk.Frame(self.nb)
        self.tab_prot = ttk.Frame(self.nb)
        self.tab_ctrl = ttk.Frame(self.nb)
        self.tab_harm = ttk.Frame(self.nb)
        self.tab_thermal = ttk.Frame(self.nb)

        self.nb.add(self.tab_main, text="Main Menu + Sizing")
        self.nb.add(self.tab_model, text="Modeling & Simulation")
        self.nb.add(self.tab_fault, text="Fault Current")
        self.nb.add(self.tab_prot, text="Protection Coordination")
        self.nb.add(self.tab_ctrl, text="Speed Control (PID/Fuzzy)")
        self.nb.add(self.tab_harm, text="Harmonics & Power Quality")
        self.nb.add(self.tab_thermal, text="Thermal + Economic")

        self._build_main_tab()
        self._build_model_tab()
        self._build_fault_tab()
        self._build_protection_tab()
        self._build_control_tab()
        self._build_harmonic_tab()
        self._build_thermal_tab()

    def _build_main_tab(self):
        left = ttk.Frame(self.tab_main)
        left.pack(side="left", fill="y", padx=8, pady=8)
        right = ttk.Frame(self.tab_main)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.p_out_mw = tk.DoubleVar(value=14.0)
        self.eff = tk.DoubleVar(value=97.3)
        self.pf = tk.DoubleVar(value=1.0)
        self.v_ll = tk.DoubleVar(value=6600.0)
        self.p_field_kw = tk.DoubleVar(value=84.0)
        self.windings = tk.IntVar(value=3)

        self._slider(left, "Output power MW", self.p_out_mw, 1, 30)
        self._slider(left, "Efficiency %", self.eff, 80, 99.5)
        self._slider(left, "Power factor", self.pf, 0.6, 1.0, 0.01)
        self._slider(left, "Line-line voltage V", self.v_ll, 400, 15000)
        self._slider(left, "DC field loss kW", self.p_field_kw, 0, 500)
        self._slider(left, "Parallel stator windings", self.windings, 1, 6, 1, is_int=True)

        bframe = ttk.Frame(left)
        bframe.pack(fill="x", pady=8)
        ttk.Button(bframe, text="Start", command=self.start).pack(side="left", padx=4)
        ttk.Button(bframe, text="Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Button(bframe, text="Reset", command=self.reset).pack(side="left", padx=4)

        self.text = tk.Text(right, height=22, wrap="word")
        self.text.pack(fill="both", expand=True)
        self._write_main_results()

        fig = Figure(figsize=(7, 4), dpi=100)
        self.ax_main = fig.add_subplot(111)
        self.canvas_main = FigureCanvasTkAgg(fig, master=right)
        self.canvas_main.get_tk_widget().pack(fill="both", expand=True)
        self._plot_current_breakdown()

    def _slider(self, parent, label, var, mn, mx, step=0.1, is_int=False):
        frm = ttk.Frame(parent)
        frm.pack(fill="x", pady=3)
        ttk.Label(frm, text=label).pack(anchor="w")
        scale = tk.Scale(frm, from_=mn, to=mx, resolution=step, orient="horizontal", variable=var, length=320,
                         command=lambda _e: self._on_param_change(), showvalue=True)
        scale.pack(fill="x")
        if is_int:
            var.trace_add("write", lambda *_: var.set(int(var.get())))

    def _compute_currents(self):
        p_out = self.p_out_mw.get() * 1e6
        eta = max(1e-4, self.eff.get() / 100.0)
        pf = max(0.05, self.pf.get())
        vll = max(10.0, self.v_ll.get())
        p_field = self.p_field_kw.get() * 1e3
        p_in = p_out / eta
        p_stator = p_in - p_field
        i_line = p_stator / (math.sqrt(3) * vll * pf)
        i_per_w = i_line / max(1, self.windings.get())
        i_thy_peak = math.sqrt(2) * i_line
        return p_in, p_stator, i_line, i_per_w, i_thy_peak

    def _write_main_results(self):
        p_in, p_st, i_line, i_w, i_peak = self._compute_currents()
        txt = (
            "Given:\n"
            "- Motor output: 14 MW (default)\n"
            "- Full-load efficiency: 97.3%\n"
            "- Unity power factor\n"
            "- DC field loss: 84 kW\n\n"
            "Formulas:\n"
            "1) P_in = P_out / eta\n"
            "2) P_stator = P_in - P_field\n"
            "3) I_stator = P_stator / (sqrt(3) * V_LL * pf)\n"
            "4) I_per_winding = I_stator / N_parallel\n"
            "5) I_thy_peak ≈ sqrt(2) * I_stator\n\n"
            f"Calculated values:\n"
            f"- Input power P_in = {p_in/1e6:.4f} MW\n"
            f"- Stator electrical power = {p_st/1e6:.4f} MW\n"
            f"- Nominal stator current (line) = {i_line:.2f} A\n"
            f"- Nominal current per winding = {i_w:.2f} A\n"
            f"- Peak thyristor current = {i_peak:.2f} A\n\n"
            "Interpretation:\n"
            "Increasing V_LL lowers current stress; lower efficiency or lower PF increases current and thermal loading."
        )
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt)

    def _plot_current_breakdown(self):
        _, _, i_line, i_w, i_peak = self._compute_currents()
        self.ax_main.clear()
        names = ["Line current", "Per winding", "Thyristor peak"]
        vals = [i_line, i_w, i_peak]
        self.ax_main.bar(names, vals, color=["#2a9d8f", "#e9c46a", "#e76f51"])
        self.ax_main.set_ylabel("A")
        self.ax_main.set_title("Current levels")
        self.ax_main.grid(True, alpha=0.3)
        self.canvas_main.draw_idle()

    def _build_model_tab(self):
        self.model_fig = Figure(figsize=(7, 4), dpi=100)
        self.model_ax = self.model_fig.add_subplot(111)
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, master=self.tab_model)
        self.model_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _simulate_motor(self):
        t = np.linspace(0, 5, 1000)
        J, B, Kt, Ke, R, L = 120.0, 2.0, 80.0, 80.0, 0.15, 0.012
        v = 3000.0
        tl = np.where(t > 2.0, 6000.0, 3500.0)
        w = np.zeros_like(t)
        i = np.zeros_like(t)
        dt = t[1] - t[0]
        for k in range(1, len(t)):
            di = (v - R * i[k-1] - Ke * w[k-1]) / L
            dw = (Kt * i[k-1] - B * w[k-1] - tl[k-1]) / J
            i[k] = np.clip(i[k-1] + dt * di, -2e4, 2e4)
            w[k] = np.clip(w[k-1] + dt * dw, 0, 500)
        return t, w, i

    def _update_sim_plot(self):
        self.sim_t, self.sim_w, self.sim_i = self._simulate_motor()
        self.model_ax.clear()
        self.model_ax.plot(self.sim_t, self.sim_w, label="Speed rad/s")
        self.model_ax.plot(self.sim_t, self.sim_i, label="Current A")
        self.model_ax.set_title("ODE Simulation")
        self.model_ax.grid(True, alpha=0.3)
        self.model_ax.legend()
        self.model_canvas.draw_idle()

    def _build_fault_tab(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        t = np.linspace(0, 0.2, 700)
        i = 20000*np.sin(2*np.pi*60*t)*np.exp(-18*t)
        ax.plot(t, i, color="crimson")
        ax.set_title("Fault current transient")
        ax.set_xlabel("s")
        ax.set_ylabel("A")
        ax.grid(True, alpha=0.3)
        c = FigureCanvasTkAgg(fig, master=self.tab_fault)
        c.get_tk_widget().pack(fill="both", expand=True)

    def _build_protection_tab(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        m = np.linspace(1.1, 20, 400)
        t_relay = 0.14 / (np.power(m, 0.02) - 1)
        t_fuse = 1.2 / np.power(m, 1.7)
        ax.loglog(m, t_relay, label="IDMT relay")
        ax.loglog(m, t_fuse, label="Fuse")
        ax.set_title("Protection coordination")
        ax.set_xlabel("Multiple of pickup")
        ax.set_ylabel("Trip time s")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        c = FigureCanvasTkAgg(fig, master=self.tab_prot)
        c.get_tk_widget().pack(fill="both", expand=True)

    def _build_control_tab(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        t = np.linspace(0, 6, 1200)
        ref = 1.0
        y_pid = ref * (1 - np.exp(-1.4*t) * (np.cos(3*t) + 0.4*np.sin(3*t)))
        y_fuzzy = ref * (1 - np.exp(-1.7*t) * (np.cos(2.4*t) + 0.22*np.sin(2.4*t)))
        load = np.where(t > 3, -0.12, 0)
        ax.plot(t, y_pid + load, label="PID")
        ax.plot(t, y_fuzzy + 0.5*load, label="Fuzzy")
        ax.axvline(3, ls="--", color="k", alpha=0.4, label="Step load")
        ax.set_title("Speed control with step load")
        ax.grid(True, alpha=0.3)
        ax.legend()
        c = FigureCanvasTkAgg(fig, master=self.tab_ctrl)
        c.get_tk_widget().pack(fill="both", expand=True)

    def _build_harmonic_tab(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        h = np.arange(1, 26, 2)
        mag = 100 / h
        thd = math.sqrt(np.sum((mag[1:]**2))) / mag[0] * 100
        ax.stem(h, mag, basefmt=" ")
        ax.set_title(f"Harmonic spectrum (THD ≈ {thd:.2f}%)")
        ax.set_xlabel("Harmonic order")
        ax.set_ylabel("% fundamental")
        ax.grid(True, alpha=0.3)
        c = FigureCanvasTkAgg(fig, master=self.tab_harm)
        c.get_tk_widget().pack(fill="both", expand=True)

    def _build_thermal_tab(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        hours = np.arange(0, 24)
        temp = 40 + 25 * (1 - np.exp(-hours/6))
        cost = 0.12 * (self.p_out_mw.get()*1e3) * hours
        ax.plot(hours, temp, label="Winding temp °C")
        ax2 = ax.twinx()
        ax2.plot(hours, cost, color="tab:orange", label="Energy cost $")
        ax.set_title("Thermal and economic trend")
        ax.set_xlabel("Hours")
        ax.grid(True, alpha=0.3)
        c = FigureCanvasTkAgg(fig, master=self.tab_thermal)
        c.get_tk_widget().pack(fill="both", expand=True)

    def _on_param_change(self):
        self._write_main_results()
        self._plot_current_breakdown()

    def start(self):
        self.running = True
        self._update_sim_plot()

    def stop(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def reset(self):
        self.stop()
        self.p_out_mw.set(14.0)
        self.eff.set(97.3)
        self.pf.set(1.0)
        self.v_ll.set(6600.0)
        self.p_field_kw.set(84.0)
        self.windings.set(3)
        self._on_param_change()

    def _on_resize(self, _event):
        for c in (self.canvas_main, self.model_canvas):
            c.draw_idle()


if __name__ == "__main__":
    app = MotorApp()
    app.mainloop()
