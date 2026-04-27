#!/usr/bin/env python3
"""
DC Motor 4-Quadrant Converter — Engineering Analysis Lab
=========================================================
200 hp, 250 V, 600 rpm DC Motor | GTO 4-Quadrant H-Bridge | f_sw = 125 Hz

Tabs:
  0  Problem Statement + H-Bridge Schematic
  1  Parameters & Calculations  (parts a-d)
  2  Transient ODE Simulation
  3  Fault Current Simulation
  4  Protection Coordination (IDMT curves)
  5  Speed Controller  (PID + Fuzzy)
  6  Thermal & Economic Analysis
  7  Harmonic & Power Quality
  8  Comprehensive Analysis
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from scipy.integrate import solve_ivp
import threading

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
import matplotlib.patches as mp
import matplotlib.lines as mlines

# ── Colour Theme ──────────────────────────────────────────────────────────────
BG     = '#06090f'
BG2    = '#0c1220'
CARD   = '#121a2c'
BORDER = '#243352'
ACC    = '#00e5ff'
ACC2   = '#7c5cff'
ACC3   = '#ffb020'
ACC4   = '#10d18d'
ACC5   = '#ff5577'
TXT    = '#e2e8f0'
DIM    = '#94a3b8'

# ── Default Motor / Converter Parameters ─────────────────────────────────────
DEF = dict(
    Vs      = 280.0,    # DC source voltage [V]
    Vn      = 250.0,    # Motor rated voltage [V]
    Ra      = 0.012,    # Armature resistance [Ω]
    La      = 350e-6,   # Armature inductance [H]
    Ia      = 620.0,    # Rated armature current [A]
    n_rated = 600.0,    # Rated speed [rpm]
    P_hp    = 200.0,    # Rated power [hp]
    f_sw    = 125.0,    # Switching frequency [Hz]
    Vdrop   = 2.0,      # Switch voltage drop [V/device]
    J       = 15.0,     # Rotor inertia [kg·m²]
    Bf      = 0.05,     # Viscous friction [N·m·s/rad]
)

# ── Physical helpers ─────────────────────────────────────────────────────────

def ke_from_p(p):
    """Back-EMF / torque constant [V·s/rad = N·m/A]."""
    omega_r = p['n_rated'] * 2.0 * np.pi / 60.0
    return (p['Vn'] - p['Ia'] * p['Ra']) / omega_r


def rated_torque(p):
    """Rated shaft torque [N·m]."""
    omega_r = p['n_rated'] * 2.0 * np.pi / 60.0
    return p['P_hp'] * 745.7 / omega_r


def duty_ripple(p, drop=False, bipolar=False):
    """
    Compute duty cycle D and peak-to-peak current ripple for stalled armature.
    Returns (D, ripple_A, V_on, V_off)
    """
    Vs  = p['Vs']
    Ra  = p['Ra']
    La  = p['La']
    Ia  = p['Ia']
    f   = p['f_sw']
    Vd  = p['Vdrop'] if drop else 0.0

    if bipolar:
        V_on  =  Vs - 2.0 * Vd
        V_off = -(Vs - 2.0 * Vd)
    else:
        V_on  = Vs - 2.0 * Vd    # Q1 + Q4 conducting
        V_off = -2.0 * Vd        # freewheeling D2 + Q4

    dV = V_on - V_off
    dV = dV if abs(dV) > 1e-12 else 1e-12

    # Average voltage = Ia*Ra (stalled, back-EMF = 0)
    D = (Ia * Ra - V_off) / dV
    D = float(np.clip(D, 1e-4, 1 - 1e-4))

    ripple = dV * D * (1.0 - D) / (La * f)
    return D, ripple, V_on, V_off


# ── Fuzzy controller helpers ─────────────────────────────────────────────────
_FS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])   # NB NS ZE PS PB centres

_FRULES = np.array([                            # rows=e, cols=de, value=out-set
    [0, 0, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [1, 2, 2, 3, 4],
    [2, 3, 3, 4, 4],
    [3, 4, 4, 4, 4],
])


def _tri(x, a, b, c):
    left  = (x - a) / (b - a + 1e-12)
    right = (c - x) / (c - b + 1e-12)
    return float(np.clip(min(left, right), 0.0, 1.0))


def _fuzzy_delta_D(err_n, derr_n):
    """Mamdani fuzzy: normalised error → normalised ΔD output."""
    mu_e  = [_tri(err_n,  _FS[max(i-1,0)], _FS[i], _FS[min(i+1,4)]) for i in range(5)]
    mu_de = [_tri(derr_n, _FS[max(j-1,0)], _FS[j], _FS[min(j+1,4)]) for j in range(5)]
    u = np.linspace(-1.0, 1.0, 101)
    agg = np.zeros(101)
    for i in range(5):
        for j in range(5):
            s = min(mu_e[i], mu_de[j])
            if s < 1e-6:
                continue
            c = _FS[_FRULES[i, j]]
            mf = np.array([_tri(x, c - 0.5, c, c + 0.5) for x in u])
            agg = np.maximum(agg, s * mf)
    tot = agg.sum()
    return 0.0 if tot < 1e-10 else float(np.dot(u, agg) / tot)


# ═════════════════════════════════════════════════════════════════════════════
#  App
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.p = dict(DEF)
        self._stop_flag = threading.Event()
        self.title("DC Motor 4-Quadrant Converter — Engineering Analysis Lab")
        try:
            self.state('zoomed')
        except Exception:
            self.geometry('1440x900')
        self.configure(bg=BG)
        self.minsize(1000, 680)

        self._setup_style()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        self._build_tab0()
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        self._build_tab5()
        self._build_tab6()
        self._build_tab7()
        self._build_tab8()

        self._set_status("Ready  |  200 hp  250 V  600 rpm  DC Motor Lab  |  Ra=12 mΩ  La=350 µH  Ia=620 A")

    # ── Style ─────────────────────────────────────────────────────────────────

    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use('default')
        s.configure('TNotebook',        background=BG,   borderwidth=0)
        s.configure('TNotebook.Tab',    background=CARD, foreground=TXT,
                     padding=[10, 5],  font=('Consolas', 9, 'bold'))
        s.map('TNotebook.Tab',
              background=[('selected', ACC2)],
              foreground=[('selected', '#ffffff')])
        s.configure('TFrame',       background=BG)
        s.configure('TLabel',       background=BG,   foreground=TXT,
                     font=('Consolas', 9))
        s.configure('TButton',      background=CARD, foreground=ACC,
                     font=('Consolas', 9, 'bold'), borderwidth=1, relief='flat',
                     padding=[8, 4])
        s.map('TButton', background=[('active', ACC2)], foreground=[('active', '#fff')])
        s.configure('Accent.TButton', background=ACC2, foreground='#fff',
                     font=('Consolas', 9, 'bold'), borderwidth=0, relief='flat',
                     padding=[10, 5])
        s.map('Accent.TButton', background=[('active', ACC)])
        s.configure('TScale',       background=BG, troughcolor=CARD, sliderlength=14)
        s.configure('TRadiobutton', background=BG, foreground=TXT, font=('Consolas', 9))
        s.configure('TCheckbutton', background=BG, foreground=TXT, font=('Consolas', 9))

    # ── Matplotlib dark theme ──────────────────────────────────────────────────

    def _mpl_dark(self, fig):
        fig.patch.set_facecolor(CARD)
        for ax in fig.axes:
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TXT, labelsize=7)
            ax.xaxis.label.set_color(TXT)
            ax.yaxis.label.set_color(TXT)
            ax.title.set_color(ACC)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        h = tk.Frame(self, bg='#0a1226', height=40)
        h.pack(fill='x', side='top')
        h.pack_propagate(False)
        tk.Label(h,
                 text='⚡  DC MOTOR 4-QUADRANT CONVERTER  —  200 hp / 250 V / 600 rpm  —  GTO H-BRIDGE  —  f_sw=125 Hz',
                 bg='#0a1226', fg=ACC,
                 font=('Consolas', 10, 'bold')).pack(side='left', padx=14)
        tk.Label(h,
                 text='Ra=12 mΩ  |  La=350 µH  |  Ia_rated=620 A  |  Vs=280 V',
                 bg='#0a1226', fg=DIM,
                 font=('Consolas', 8)).pack(side='right', padx=14)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=CARD, height=22)
        sb.pack(fill='x', side='bottom')
        sb.pack_propagate(False)
        self._sv = tk.StringVar(value='Ready')
        tk.Label(sb, textvariable=self._sv, bg=CARD, fg=ACC4,
                 font=('Consolas', 8), anchor='w').pack(side='left', padx=8)

    def _set_status(self, msg):
        self._sv.set(msg)

    # ── Notebook ──────────────────────────────────────────────────────────────

    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=3, pady=3)
        for title, attr in [
            ('📋 Problem',      't0'),
            ('⚙ Parameters',   't1'),
            ('📈 Simulation',   't2'),
            ('⚡ Fault',        't3'),
            ('🛡 Protection',   't4'),
            ('🎛 Speed Ctrl',   't5'),
            ('🌡 Thermal/Eco',  't6'),
            ('〜 Harmonics',    't7'),
            ('📊 Comprehensive','t8'),
        ]:
            f = ttk.Frame(self.nb)
            setattr(self, attr, f)
            self.nb.add(f, text=title)

    # ── Reusable widget helpers ───────────────────────────────────────────────

    def _scroll_left(self, parent, width=300):
        """Create a scrollable left-panel frame; returns the inner frame."""
        outer = tk.Frame(parent, bg=BG, width=width)
        outer.pack(side='left', fill='y', padx=2, pady=2)
        outer.pack_propagate(False)
        cvs = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb  = ttk.Scrollbar(outer, orient='vertical', command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cvs.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(cvs, bg=BG)
        win_id = cvs.create_window((0, 0), window=inner, anchor='nw')

        def _resize_inner(ev):
            cvs.itemconfig(win_id, width=ev.width)
        cvs.bind('<Configure>', _resize_inner)

        def _update_scroll(ev):
            cvs.configure(scrollregion=cvs.bbox('all'))
        inner.bind('<Configure>', _update_scroll)

        def _mouse_wheel(ev):
            cvs.yview_scroll(int(-1 * (ev.delta / 120)), 'units')
        cvs.bind_all('<MouseWheel>', _mouse_wheel)
        return inner

    def _make_fig(self, parent, nrows=1, ncols=1, figsize=(9, 5), toolbar=True):
        """Embed a matplotlib Figure in parent. Returns (fig, axes, canvas)."""
        fig = Figure(figsize=figsize, tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        if toolbar:
            tb = NavigationToolbar2Tk(canvas, parent)
            tb.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        parent.bind('<Configure>', lambda e: (fig.tight_layout(pad=0.4), canvas.draw_idle()))
        axes = fig.subplots(nrows, ncols)
        if nrows == 1 and ncols == 1:
            axes = axes
        return fig, axes, canvas

    def _slider(self, parent, label, var, lo, hi, digits=3, cmd=None):
        """Create label + Scale + value display in parent. Returns (frame, var)."""
        row = tk.Frame(parent, bg=BG)
        row.pack(fill='x', padx=6, pady=2)
        tk.Label(row, text=label, bg=BG, fg=TXT, font=('Consolas', 8),
                 anchor='w', width=22).pack(side='left')
        sc = ttk.Scale(row, variable=var, from_=lo, to=hi, orient='horizontal',
                       length=130, command=lambda v: (var.set(round(float(v), digits)),
                                                       lbl.config(text=f'{float(var.get()):.{digits}g}'),
                                                       cmd() if cmd else None))
        sc.pack(side='left', padx=4)
        lbl = tk.Label(row, text=f'{float(var.get()):.{digits}g}',
                       bg=BG, fg=ACC3, font=('Consolas', 8), width=10)
        lbl.pack(side='left')
        # Keep label in sync even when var is set programmatically
        var.trace_add('write', lambda *a: lbl.config(text=f'{float(var.get()):.{digits}g}'))
        return row

    def _btn(self, parent, text, cmd, style='Accent.TButton'):
        b = ttk.Button(parent, text=text, command=cmd, style=style)
        b.pack(side='left', padx=4, pady=4)
        return b

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=6, pady=4)

    def _lbl_h(self, parent, text):
        tk.Label(parent, text=text, bg=BG, fg=ACC,
                 font=('Consolas', 9, 'bold')).pack(anchor='w', padx=6, pady=(6, 1))

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 0 — Problem Statement + H-Bridge Schematic
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab0(self):
        # Left: problem text
        lf = tk.Frame(self.t0, bg=BG)
        lf.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        tk.Label(lf, text='Problem Statement — 200 hp DC Motor 4-Quadrant Drive',
                 bg=BG, fg=ACC, font=('Consolas', 10, 'bold')).pack(anchor='w', pady=(0, 4))

        txt = scrolledtext.ScrolledText(lf, bg=CARD, fg=TXT, font=('Consolas', 9),
                                         insertbackground=ACC, relief='flat',
                                         wrap='word', state='normal')
        txt.pack(fill='both', expand=True)
        txt.insert('end', PROBLEM_TEXT)
        txt.config(state='disabled')

        # Right: schematic
        rf = tk.Frame(self.t0, bg=BG, width=460)
        rf.pack(side='right', fill='y', padx=4, pady=4)
        rf.pack_propagate(False)

        fig0 = Figure(figsize=(4.4, 5.5))
        c0 = FigureCanvasTkAgg(fig0, master=rf)
        c0.get_tk_widget().pack(fill='both', expand=True)
        self._draw_schematic(fig0)
        c0.draw()

    def _draw_schematic(self, fig):
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG2)
        fig.patch.set_facecolor(CARD)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('4-Quadrant H-Bridge (Startup: Q3 open, Q4 closed)',
                     color=ACC, fontsize=8, pad=6)

        # Bus bars
        ax.plot([1, 9], [10.5, 10.5], color=ACC3, lw=2.5)   # + rail
        ax.plot([1, 9], [1.5,  1.5],  color=ACC5, lw=2.5)   # - rail
        ax.text(0.3, 10.5, '+Vs', color=ACC3, fontsize=8, va='center')
        ax.text(0.3, 1.5,  '−',  color=ACC5, fontsize=8, va='center')

        # Left leg (Q1 top, Q2 bottom)
        ax.plot([2, 2], [1.5, 10.5], color=BORDER, lw=1.5, ls='--')
        self._gto(ax, 2, 8.5, 'Q1\n(PWM)', ACC4)
        self._gto(ax, 2, 3.5, 'Q2\n(OFF)', DIM)

        # Right leg (Q3 top, Q4 bottom)
        ax.plot([8, 8], [1.5, 10.5], color=BORDER, lw=1.5, ls='--')
        self._gto(ax, 8, 8.5, 'Q3\n(open)', ACC5)
        self._gto(ax, 8, 3.5, 'Q4\n(ON)',  ACC4)

        # Motor in middle
        motor_rect = mpatches.FancyBboxPatch((3.8, 5.0), 2.4, 2.0,
                                              boxstyle='round,pad=0.1',
                                              fc=CARD, ec=ACC2, lw=1.5)
        ax.add_patch(motor_rect)
        ax.text(5.0, 6.0, 'M\n(Stalled)', color=ACC2, fontsize=8,
                ha='center', va='center', fontweight='bold')

        # Connections motor to legs
        ax.plot([2, 3.8], [6.0, 6.0],  color=TXT, lw=1.5)
        ax.plot([6.2, 8], [6.0, 6.0],  color=TXT, lw=1.5)

        # Freewheeling path annotation
        ax.annotate('', xy=(2, 1.5), xytext=(2, 6),
                    arrowprops=dict(arrowstyle='->', color=ACC3,
                                    connectionstyle='arc3,rad=-0.4', lw=1.5))
        ax.text(0.3, 4.5, 'Free-\nwheel\npath', color=ACC3, fontsize=6.5, ha='center')

        # Labels
        ax.text(5.0, 11.2, 'Vdc = 280 V', color=TXT, fontsize=8, ha='center')
        ax.text(5.0, 0.5,
                'Motor: Ra=12mΩ  La=350µH  Ia_rated=620A',
                color=DIM, fontsize=6.5, ha='center')

    def _gto(self, ax, x, y, label, color):
        tri = mpatches.RegularPolygon((x, y), 3, radius=0.55,
                                       orientation=0, fc=color, ec='#fff',
                                       alpha=0.85, lw=1)
        ax.add_patch(tri)
        ax.text(x + 0.85, y, label, color=color, fontsize=6.5, va='center')

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Parameters & Calculations
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab1(self):
        left = self._scroll_left(self.t1, width=310)
        right = tk.Frame(self.t1, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        # ── sliders ──
        self._lbl_h(left, '─ Converter / Source')
        self._v_Vs    = tk.DoubleVar(value=DEF['Vs'])
        self._v_Vdrop = tk.DoubleVar(value=DEF['Vdrop'])
        self._v_fsw   = tk.DoubleVar(value=DEF['f_sw'])
        self._slider(left, 'Vs  [V]',       self._v_Vs,    200, 400, 1, self._do_calc)
        self._slider(left, 'Vdrop [V/sw]',  self._v_Vdrop, 0,   5,   2, self._do_calc)
        self._slider(left, 'f_sw  [Hz]',    self._v_fsw,   50,  500, 1, self._do_calc)

        self._lbl_h(left, '─ Motor Armature')
        self._v_Ra  = tk.DoubleVar(value=DEF['Ra'] * 1000)   # mΩ display
        self._v_La  = tk.DoubleVar(value=DEF['La'] * 1e6)    # µH display
        self._v_Ia  = tk.DoubleVar(value=DEF['Ia'])
        self._slider(left, 'Ra  [mΩ]',  self._v_Ra, 1,   100, 2, self._do_calc)
        self._slider(left, 'La  [µH]',  self._v_La, 50,  2000, 1, self._do_calc)
        self._slider(left, 'Ia  [A]',   self._v_Ia, 100, 1500, 1, self._do_calc)

        self._lbl_h(left, '─ Motor Nameplate')
        self._v_Vn     = tk.DoubleVar(value=DEF['Vn'])
        self._v_nr     = tk.DoubleVar(value=DEF['n_rated'])
        self._v_Php    = tk.DoubleVar(value=DEF['P_hp'])
        self._slider(left, 'Vn  [V]',       self._v_Vn,  100, 600, 1, self._do_calc)
        self._slider(left, 'n_rated [rpm]', self._v_nr,  100, 3000, 1, self._do_calc)
        self._slider(left, 'P  [hp]',       self._v_Php, 50,  1000, 1, self._do_calc)

        self._lbl_h(left, '─ Mechanical')
        self._v_J  = tk.DoubleVar(value=DEF['J'])
        self._v_Bf = tk.DoubleVar(value=DEF['Bf'])
        self._slider(left, 'J  [kg·m²]', self._v_J,  0.5, 100, 2, self._do_calc)
        self._slider(left, 'Bf [N·m·s]', self._v_Bf, 0.001, 2.0, 3, self._do_calc)

        self._sep(left)
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill='x', padx=6, pady=4)
        self._btn(brow, '▶ Calculate',     self._do_calc)
        self._btn(brow, '↺ Reset Defaults', self._reset_defaults1)

        # ── right: results text + chart ──
        res_frame = tk.Frame(right, bg=BG)
        res_frame.pack(fill='both', expand=True)

        # Top: text
        self._t1_text = scrolledtext.ScrolledText(
            res_frame, bg=CARD, fg=TXT, font=('Consolas', 9),
            height=18, relief='flat', wrap='word')
        self._t1_text.pack(fill='x', padx=4, pady=(4, 0))

        # Bottom: chart
        chart_frame = tk.Frame(res_frame, bg=BG)
        chart_frame.pack(fill='both', expand=True, padx=4, pady=4)
        self._t1_fig, self._t1_axes, self._t1_canvas = self._make_fig(
            chart_frame, nrows=1, ncols=2, figsize=(9, 3.2))

        self._do_calc()

    def _p_from_sliders1(self):
        return dict(
            Vs      = float(self._v_Vs.get()),
            Vn      = float(self._v_Vn.get()),
            Ra      = float(self._v_Ra.get()) / 1000.0,
            La      = float(self._v_La.get()) / 1e6,
            Ia      = float(self._v_Ia.get()),
            n_rated = float(self._v_nr.get()),
            P_hp    = float(self._v_Php.get()),
            f_sw    = float(self._v_fsw.get()),
            Vdrop   = float(self._v_Vdrop.get()),
            J       = float(self._v_J.get()),
            Bf      = float(self._v_Bf.get()),
        )

    def _do_calc(self, *_):
        p = self._p_from_sliders1()
        self.p.update(p)    # keep global params in sync

        Ke = ke_from_p(p)
        Tr = rated_torque(p)
        P_W = p['P_hp'] * 745.7

        Da, Ra_pp, Von_a, Voff_a = duty_ripple(p, drop=False, bipolar=False)
        Db, Rb_pp, Von_b, Voff_b = duty_ripple(p, drop=False, bipolar=False)   # same as a
        Dc, Rc_pp, Von_c, Voff_c = duty_ripple(p, drop=True,  bipolar=False)
        Dd, Rd_pp, Von_d, Voff_d = duty_ripple(p, drop=True,  bipolar=True)

        omega_r = p['n_rated'] * 2.0 * np.pi / 60.0
        tel_a = p['La'] / p['Ra']
        I_ratio = Rd_pp / (Ra_pp + 1e-12)

        lines = [
            '═' * 64,
            '  200 hp DC Motor — 4-Quadrant GTO Converter Analysis',
            '═' * 64,
            '',
            '  MOTOR PARAMETERS',
            f'  Ra  = {p["Ra"]*1000:.3f} mΩ  ({p["Ra"]:.5f} Ω)',
            f'  La  = {p["La"]*1e6:.1f} µH  ({p["La"]:.3e} H)',
            f'  Ia  = {p["Ia"]:.1f} A  (rated)',
            f'  Vn  = {p["Vn"]:.1f} V  (rated)',
            f'  Vs  = {p["Vs"]:.1f} V  (dc source)',
            f'  ωr  = {omega_r:.4f} rad/s  ({p["n_rated"]:.0f} rpm)',
            f'  Ke  = Kt = {Ke:.4f} V·s/rad',
            f'  Tr  = {Tr:.1f} N·m  ({p["P_hp"]:.0f} hp = {P_W:.0f} W)',
            f'  τe  = La/Ra = {tel_a*1000:.3f} ms',
            '',
            '─' * 64,
            '  (a) DUTY CYCLE — Ideal switches, stalled armature',
            '─' * 64,
            '  Back-EMF E = 0 (ω = 0)',
            '  Unipolar 2-quadrant:  V_on = Vs,  V_off = 0',
            f'  Average voltage = D·Vs = Ia·Ra',
            f'  D = Ia·Ra / Vs = {p["Ia"]:.1f}×{p["Ra"]:.5f} / {p["Vs"]:.1f}',
            f'  D = {Da*100:.4f} %   ({Da:.6f})',
            '',
            '─' * 64,
            '  (b) PEAK-TO-PEAK RIPPLE — Ideal switches',
            '─' * 64,
            '  ΔI = (V_on − V_off) · D · (1−D) / (La · f_sw)',
            f'  ΔI = {Von_a:.2f} × {Da:.5f} × {1-Da:.5f} / ({p["La"]:.3e} × {p["f_sw"]:.0f})',
            f'  ΔI = {Ra_pp:.2f} A  (peak-to-peak)',
            '',
            '─' * 64,
            '  (c) WITH SWITCH DROPS (Vdrop = {:.1f} V / device)'.format(p['Vdrop']),
            '─' * 64,
            '  V_on  = Vs − 2·Vdrop = {:.2f} − {:.1f} = {:.2f} V'.format(
                p['Vs'], 2*p['Vdrop'], Von_c),
            '  V_off = − 2·Vdrop = {:.2f} V'.format(Voff_c),
            '  D·V_on + (1−D)·V_off = Ia·Ra',
            f'  D = (Ia·Ra − V_off) / (V_on − V_off)',
            f'    = ({p["Ia"]*p["Ra"]:.4f} − ({Voff_c:.3f})) / ({Von_c:.2f} − ({Voff_c:.3f}))',
            f'  D = {Dc*100:.4f} %   ({Dc:.6f})',
            f'  ΔI = {Rc_pp:.2f} A  (peak-to-peak)',
            '',
            '─' * 64,
            '  (d) 4-QUADRANT BIPOLAR MODE',
            '─' * 64,
            '  V_on  = +(Vs−2·Vdrop) = {:.2f} V'.format(Von_d),
            '  V_off = −(Vs−2·Vdrop) = {:.2f} V'.format(Voff_d),
            f'  D = {Dd*100:.4f} %',
            f'  ΔI = {Rd_pp:.1f} A  (peak-to-peak)',
            f'  Ratio vs case (b) ideal: {I_ratio:.1f}×  → SERIOUSLY affected',
            '',
            '─' * 64,
            '  SUMMARY TABLE',
            '─' * 64,
            '  Case   Condition          D [%]    ΔI [A]',
            f'  (a)    Ideal, 2-quad    {Da*100:7.3f}   {Ra_pp:8.2f}',
            f'  (c)    Drops, 2-quad    {Dc*100:7.3f}   {Rc_pp:8.2f}',
            f'  (d)    Drops, 4-quad    {Dd*100:7.3f}   {Rd_pp:8.1f}',
            '═' * 64,
        ]

        self._t1_text.config(state='normal')
        self._t1_text.delete('1.0', 'end')
        self._t1_text.insert('end', '\n'.join(lines))
        self._t1_text.config(state='disabled')

        # ── bar charts ──
        fig = self._t1_fig
        for ax in fig.axes:
            ax.cla()

        ax1, ax2 = self._t1_axes

        labels = ['(a) Ideal', '(c) w/drops', '(d) 4-quad']
        duties  = [Da*100, Dc*100, Dd*100]
        ripples = [Ra_pp, Rc_pp, Rd_pp]
        colors  = [ACC4, ACC3, ACC5]

        ax1.bar(labels, duties,  color=colors, edgecolor=BORDER, linewidth=0.8)
        ax1.set_ylabel('Duty Cycle [%]', color=TXT)
        ax1.set_title('Duty Cycle Comparison', color=ACC)
        for i, v in enumerate(duties):
            ax1.text(i, v + 0.002 * max(duties), f'{v:.3f}%',
                     ha='center', color=TXT, fontsize=7.5)

        ax2.bar(labels, ripples, color=colors, edgecolor=BORDER, linewidth=0.8)
        ax2.set_ylabel('ΔI peak-to-peak [A]', color=TXT)
        ax2.set_title('Current Ripple Comparison', color=ACC)
        for i, v in enumerate(ripples):
            ax2.text(i, v + 0.01 * max(ripples), f'{v:.1f} A',
                     ha='center', color=TXT, fontsize=7.5)

        self._mpl_dark(fig)
        fig.tight_layout(pad=0.5)
        self._t1_canvas.draw_idle()
        self._set_status(f'Calculated: D={Da*100:.3f}%  ΔI={Ra_pp:.1f} A  (ideal)')

    def _reset_defaults1(self):
        self._v_Vs.set(DEF['Vs'])
        self._v_Vn.set(DEF['Vn'])
        self._v_Ra.set(DEF['Ra'] * 1000)
        self._v_La.set(DEF['La'] * 1e6)
        self._v_Ia.set(DEF['Ia'])
        self._v_nr.set(DEF['n_rated'])
        self._v_Php.set(DEF['P_hp'])
        self._v_fsw.set(DEF['f_sw'])
        self._v_Vdrop.set(DEF['Vdrop'])
        self._v_J.set(DEF['J'])
        self._v_Bf.set(DEF['Bf'])
        self._do_calc()

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — Transient ODE Simulation
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab2(self):
        left = self._scroll_left(self.t2, width=300)
        right = tk.Frame(self.t2, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._lbl_h(left, '─ Simulation Setup')
        self._v_sim_dur    = tk.DoubleVar(value=3.0)
        self._v_sim_tload  = tk.DoubleVar(value=0.0)
        self._slider(left, 'Duration [s]',      self._v_sim_dur,   0.5, 20.0, 2)
        self._slider(left, 'Load torque [N·m]', self._v_sim_tload, 0,   3000, 1)

        self._lbl_h(left, '─ Converter Mode')
        self._v_sim_mode  = tk.StringVar(value='2q')
        tk.Radiobutton(left, text='2-Quadrant (startup)',  variable=self._v_sim_mode,
                       value='2q',  bg=BG, fg=TXT,
                       font=('Consolas', 9), selectcolor=CARD).pack(anchor='w', padx=8)
        tk.Radiobutton(left, text='4-Quadrant (bipolar)',  variable=self._v_sim_mode,
                       value='4q',  bg=BG, fg=TXT,
                       font=('Consolas', 9), selectcolor=CARD).pack(anchor='w', padx=8)

        self._lbl_h(left, '─ Options')
        self._v_sim_drops = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text='Include switch drops',
                       variable=self._v_sim_drops,
                       bg=BG, fg=TXT, selectcolor=CARD,
                       font=('Consolas', 9)).pack(anchor='w', padx=8)

        self._lbl_h(left, '─ Initial Conditions')
        self._v_sim_ia0 = tk.DoubleVar(value=0.0)
        self._v_sim_w0  = tk.DoubleVar(value=0.0)
        self._slider(left, 'ia(0) [A]',    self._v_sim_ia0, -700, 700, 1)
        self._slider(left, 'ω(0)  [rpm]',  self._v_sim_w0,  0,   700, 1)

        self._sep(left)
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill='x', padx=6)
        self._btn(brow, '▶ Start',  self._start_sim2)
        self._btn(brow, '■ Stop',   self._stop_sim2)
        self._btn(brow, '↺ Reset',  self._reset_sim2)

        # results label
        self._t2_summary = tk.Label(left, text='', bg=BG, fg=ACC4,
                                     font=('Consolas', 8), justify='left',
                                     wraplength=280)
        self._t2_summary.pack(anchor='w', padx=6, pady=6)

        # figure
        self._t2_fig = Figure(figsize=(9, 6))
        self._t2_canvas = FigureCanvasTkAgg(self._t2_fig, master=right)
        NavigationToolbar2Tk(self._t2_canvas, right).update()
        self._t2_canvas.get_tk_widget().pack(fill='both', expand=True)
        right.bind('<Configure>', lambda e: (self._t2_fig.tight_layout(pad=0.4),
                                              self._t2_canvas.draw_idle()))
        self._init_t2_axes()

    def _init_t2_axes(self):
        fig = self._t2_fig
        fig.patch.set_facecolor(CARD)
        fig.clf()
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        ax1.set_ylabel('ia [A]',  color=TXT, fontsize=8)
        ax2.set_ylabel('ω [rpm]', color=TXT, fontsize=8)
        ax3.set_ylabel('va [V]',  color=TXT, fontsize=8)
        ax3.set_xlabel('Time [s]', color=TXT, fontsize=8)
        ax1.set_title('Transient Simulation — Startup', color=ACC, fontsize=9)
        ax1.xaxis.label.set_color(TXT);  ax1.yaxis.label.set_color(TXT)
        ax2.xaxis.label.set_color(TXT);  ax2.yaxis.label.set_color(TXT)
        ax3.xaxis.label.set_color(TXT);  ax3.yaxis.label.set_color(TXT)
        fig.tight_layout(pad=0.5)
        self._t2_canvas.draw_idle()
        return ax1, ax2, ax3

    def _start_sim2(self):
        self._stop_flag.clear()
        self._set_status('Simulation running …')
        t = threading.Thread(target=self._run_sim2_thread, daemon=True)
        t.start()

    def _stop_sim2(self):
        self._stop_flag.set()
        self._set_status('Simulation stopped')

    def _reset_sim2(self):
        self._stop_flag.set()
        self._init_t2_axes()
        self._t2_summary.config(text='')
        self._set_status('Reset')

    def _run_sim2_thread(self):
        p   = dict(self.p)
        dur = float(self._v_sim_dur.get())
        Tl  = float(self._v_sim_tload.get())
        drop = bool(self._v_sim_drops.get())
        bipolar = (self._v_sim_mode.get() == '4q')
        ia0  = float(self._v_sim_ia0.get())
        w0   = float(self._v_sim_w0.get()) * 2.0 * np.pi / 60.0

        Ke   = ke_from_p(p)
        Ra   = p['Ra'];   La = p['La'];  J = p['J'];  Bf = p['Bf']
        Vs   = p['Vs'];   f  = p['f_sw']; Vd = p['Vdrop'] if drop else 0.0
        Ia_max = p['Ia'] * 3.0
        T_sw  = 1.0 / f

        D, _, V_on, V_off = duty_ripple(p, drop=drop, bipolar=bipolar)
        t_load_step = dur / 3.0

        def v_applied(t):
            phase = (t % T_sw) / T_sw
            return V_on if phase < D else V_off

        def ode(t, y):
            if self._stop_flag.is_set():
                return [0.0, 0.0]
            ia, omega = y
            va = v_applied(t)
            dia_dt = (va - Ra * ia - Ke * omega) / La
            Tl_cur = Tl if t >= t_load_step else 0.0
            domega_dt = (Ke * ia - Tl_cur - Bf * omega) / J
            ia = float(np.clip(ia, -Ia_max, Ia_max))
            return [dia_dt, domega_dt]

        try:
            sol = solve_ivp(ode, [0.0, dur], [ia0, w0],
                            method='RK23', max_step=T_sw / 8,
                            rtol=1e-3, atol=1e-2,
                            dense_output=False)

            t_eval = sol.t
            ia_sol  = np.clip(sol.y[0], -Ia_max, Ia_max)
            w_sol   = sol.y[1]
            rpm_sol = w_sol * 60.0 / (2.0 * np.pi)
            va_sol  = np.array([v_applied(tt) for tt in t_eval])

            # summary
            rpm_fin  = float(rpm_sol[-1])
            ia_peak  = float(np.max(np.abs(ia_sol)))
            Te_fin   = float(Ke * ia_sol[-1])

            self.after(0, lambda: self._update_sim2_plot(
                t_eval, ia_sol, rpm_sol, va_sol,
                p['Ia'], p['n_rated'], D, t_load_step,
                rpm_fin, ia_peak, Te_fin))
        except Exception as exc:
            self.after(0, lambda: self._set_status(f'Sim error: {exc}'))

    def _update_sim2_plot(self, t, ia, rpm, va, Ia_r, n_r, D,
                           t_step, rpm_fin, ia_peak, Te_fin):
        ax1, ax2, ax3 = self._init_t2_axes()

        ax1.plot(t, ia,  color=ACC,  lw=1.2, label='ia(t)')
        ax1.axhline(Ia_r,  color=ACC5, lw=0.8, ls='--', label=f'Ia_rated={Ia_r:.0f}A')
        ax1.axhline(-Ia_r, color=ACC5, lw=0.8, ls='--')
        ax1.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)
        ax1.set_ylabel('ia [A]', color=TXT, fontsize=8)
        ax1.yaxis.label.set_color(TXT)

        ax2.plot(t, rpm, color=ACC4, lw=1.2, label='ω(t) [rpm]')
        ax2.axhline(n_r, color=ACC3, lw=0.8, ls='--', label=f'n_rated={n_r:.0f}rpm')
        ax2.axvline(t_step, color=DIM, lw=0.8, ls=':', label='Load step')
        ax2.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)
        ax2.set_ylabel('ω [rpm]', color=TXT, fontsize=8)
        ax2.yaxis.label.set_color(TXT)

        ax3.plot(t, va, color=ACC3, lw=0.8, label='va(t)')
        ax3.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)
        ax3.set_ylabel('va [V]', color=TXT, fontsize=8)
        ax3.set_xlabel('Time [s]', color=TXT, fontsize=8)
        ax3.yaxis.label.set_color(TXT)
        ax3.xaxis.label.set_color(TXT)

        self._t2_fig.tight_layout(pad=0.5)
        self._t2_canvas.draw_idle()

        self._t2_summary.config(
            text=(f'D = {D*100:.3f}%\n'
                  f'Final speed  = {rpm_fin:.1f} rpm\n'
                  f'Peak current = {ia_peak:.1f} A\n'
                  f'Final torque = {Te_fin:.1f} N·m'))
        self._set_status(f'Done  |  ω_final={rpm_fin:.1f} rpm  ia_peak={ia_peak:.1f} A')

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — Fault Current Simulation
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab3(self):
        left = self._scroll_left(self.t3, width=300)
        right = tk.Frame(self.t3, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._lbl_h(left, '─ Fault Parameters')
        self._v_f3_t0    = tk.DoubleVar(value=0.2)
        self._v_f3_Rfault = tk.DoubleVar(value=0.0)
        self._v_f3_fdur  = tk.DoubleVar(value=0.1)
        self._v_f3_dur   = tk.DoubleVar(value=1.5)
        self._v_f3_speed = tk.DoubleVar(value=600.0)
        self._slider(left, 'Pre-fault speed [rpm]', self._v_f3_speed, 0,   700, 1)
        self._slider(left, 'Fault onset t [s]',     self._v_f3_t0,   0.05, 1.0, 3)
        self._slider(left, 'Fault Rfault [mΩ]',     self._v_f3_Rfault, 0, 50,  2)
        self._slider(left, 'Fault duration [s]',    self._v_f3_fdur, 0.01, 0.5, 3)
        self._slider(left, 'Total sim [s]',         self._v_f3_dur,  0.5,  4.0, 2)

        self._sep(left)
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill='x', padx=6)
        self._btn(brow, '▶ Run Fault Sim', self._run_fault3)

        self._t3_info = tk.Label(left, text='', bg=BG, fg=ACC5,
                                  font=('Consolas', 8), justify='left', wraplength=280)
        self._t3_info.pack(anchor='w', padx=6, pady=8)

        # Explanation
        tk.Label(left, text=(
            'Model:\n'
            'Short-circuit at t_fault adds\n'
            'Rfault in parallel with Ra.\n'
            'Breaker trips at 2×Ia_rated.\n'
            'Post-fault: normal operation.'),
            bg=BG, fg=DIM, font=('Consolas', 8), justify='left',
            wraplength=280).pack(anchor='w', padx=6)

        self._t3_fig = Figure(figsize=(9, 5))
        self._t3_canvas = FigureCanvasTkAgg(self._t3_fig, master=right)
        NavigationToolbar2Tk(self._t3_canvas, right).update()
        self._t3_canvas.get_tk_widget().pack(fill='both', expand=True)
        right.bind('<Configure>', lambda e: (self._t3_fig.tight_layout(pad=0.4),
                                              self._t3_canvas.draw_idle()))

    def _run_fault3(self):
        self._set_status('Running fault simulation …')
        threading.Thread(target=self._fault3_thread, daemon=True).start()

    def _fault3_thread(self):
        p   = dict(self.p)
        t0  = float(self._v_f3_t0.get())
        Rf  = float(self._v_f3_Rfault.get()) / 1000.0
        fdur = float(self._v_f3_fdur.get())
        dur  = float(self._v_f3_dur.get())
        n0   = float(self._v_f3_speed.get())
        w0   = n0 * 2.0 * np.pi / 60.0

        Ke = ke_from_p(p)
        Ra = p['Ra'];  La = p['La'];  J = p['J'];  Bf = p['Bf']
        D, _, V_on, V_off = duty_ripple(p, drop=True, bipolar=False)
        T_sw = 1.0 / p['f_sw']
        Ia_max = p['Ia'] * 5.0
        trip_level = p['Ia'] * 2.0

        def v_app(t):
            return V_on if (t % T_sw) / T_sw < D else V_off

        tripped = [False]
        trip_time = [None]

        def ode(t, y):
            ia, omega = y
            if tripped[0]:
                va = 0.0; Ra_eff = Ra
            else:
                va = v_app(t)
                fault_active = (t0 <= t <= t0 + fdur)
                Ra_eff = Ra + Rf if fault_active and Rf > 0 else Ra
                # Short-circuit: Rfault ~ 0 → huge current surge
                if fault_active and Rf == 0:
                    Ra_eff = Ra * 0.01  # near-short
            dia = (va - Ra_eff * ia - Ke * omega) / La
            domega = (Ke * ia - Bf * omega) / J
            ia = float(np.clip(ia, -Ia_max, Ia_max))
            if abs(ia) >= trip_level and not tripped[0]:
                tripped[0] = True
                trip_time[0] = t
            return [dia, domega]

        ia0 = Ke * w0 / Ra   # pre-fault steady state approx
        ia0 = float(np.clip(ia0, -p['Ia'], p['Ia']))

        sol = solve_ivp(ode, [0, dur], [ia0, w0],
                        method='RK23', max_step=T_sw / 8,
                        rtol=1e-3, atol=1e-2)

        t_arr  = sol.t
        ia_arr = np.clip(sol.y[0], -Ia_max, Ia_max)
        rpm_arr = sol.y[1] * 60.0 / (2.0 * np.pi)

        peak_fault = float(np.max(np.abs(ia_arr)))
        self.after(0, lambda: self._update_fault3(
            t_arr, ia_arr, rpm_arr, t0, t0+fdur, trip_level,
            trip_time[0], peak_fault, p['Ia']))

    def _update_fault3(self, t, ia, rpm, t0, t1, trip_lv,
                        trip_t, peak, Ia_r):
        fig = self._t3_fig
        fig.patch.set_facecolor(CARD)
        fig.clf()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        for ax in (ax1, ax2):
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        ax1.plot(t, ia,  color=ACC,  lw=1.2, label='ia(t) [A]')
        ax1.axhline( trip_lv, color=ACC5, lw=1, ls='--', label=f'Trip={trip_lv:.0f}A')
        ax1.axhline(-trip_lv, color=ACC5, lw=1, ls='--')
        ax1.axvspan(t0, t1, color=ACC5, alpha=0.15, label='Fault window')
        if trip_t:
            ax1.axvline(trip_t, color=ACC3, lw=1.2, ls=':', label=f'Trip @ {trip_t:.3f}s')
        ax1.set_ylabel('Armature Current [A]', color=TXT, fontsize=8)
        ax1.yaxis.label.set_color(TXT)
        ax1.set_title('Fault Current Analysis', color=ACC, fontsize=9)
        ax1.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)

        ax2.plot(t, rpm, color=ACC4, lw=1.2, label='ω [rpm]')
        ax2.axvspan(t0, t1, color=ACC5, alpha=0.15)
        ax2.set_ylabel('Speed [rpm]', color=TXT, fontsize=8)
        ax2.set_xlabel('Time [s]',   color=TXT, fontsize=8)
        ax2.yaxis.label.set_color(TXT)
        ax2.xaxis.label.set_color(TXT)
        ax2.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)

        fig.tight_layout(pad=0.5)
        self._t3_canvas.draw_idle()

        info = (f'Peak fault current : {peak:.1f} A\n'
                f'Rated current      : {Ia_r:.1f} A\n'
                f'Trip level (2×)    : {trip_lv:.1f} A\n'
                f'Breaker trip time  : {trip_t:.4f} s' if trip_t else
                f'Peak fault current : {peak:.1f} A\n'
                f'Rated current      : {Ia_r:.1f} A\n'
                f'Trip level (2×)    : {trip_lv:.1f} A\n'
                f'No trip occurred')
        self._t3_info.config(text=info)
        self._set_status(f'Fault sim done  |  peak ia = {peak:.1f} A')

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — Protection Coordination (IDMT curves)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab4(self):
        left = self._scroll_left(self.t4, width=300)
        right = tk.Frame(self.t4, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._lbl_h(left, '─ Relay Settings')
        self._v_p4_CTr  = tk.DoubleVar(value=600.0)
        self._v_p4_TDS  = tk.DoubleVar(value=0.3)
        self._v_p4_Ip   = tk.DoubleVar(value=1.1)    # × Ia_rated
        self._v_p4_Ifault = tk.DoubleVar(value=3.0)  # × Ia_rated
        self._slider(left, 'CT ratio',          self._v_p4_CTr,   50,  2000, 1)
        self._slider(left, 'Time dial TDS',     self._v_p4_TDS,   0.05, 2.0, 2)
        self._slider(left, 'Pickup (× Ia)',     self._v_p4_Ip,    0.5,  3.0, 2)
        self._slider(left, 'Fault level (× Ia)',self._v_p4_Ifault,1.5, 15.0, 2)

        self._sep(left)
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill='x', padx=6)
        self._btn(brow, '▶ Calculate', self._run_prot4)

        self._t4_info = tk.Label(left, text='', bg=BG, fg=ACC4,
                                  font=('Consolas', 8), justify='left', wraplength=280)
        self._t4_info.pack(anchor='w', padx=6, pady=6)

        tk.Label(left, text=(
            'IEC 60255 curves:\n'
            '• Standard Inverse (SI)\n'
            '  t = TDS×0.14/((I/Ip)^0.02−1)\n'
            '• Very Inverse (VI)\n'
            '  t = TDS×13.5/((I/Ip)−1)\n'
            '• Extremely Inverse (EI)\n'
            '  t = TDS×80/((I/Ip)²−1)'),
            bg=BG, fg=DIM, font=('Consolas', 8), justify='left',
            wraplength=280).pack(anchor='w', padx=6)

        self._t4_fig = Figure(figsize=(9, 5))
        self._t4_canvas = FigureCanvasTkAgg(self._t4_fig, master=right)
        NavigationToolbar2Tk(self._t4_canvas, right).update()
        self._t4_canvas.get_tk_widget().pack(fill='both', expand=True)
        right.bind('<Configure>', lambda e: (self._t4_fig.tight_layout(pad=0.4),
                                              self._t4_canvas.draw_idle()))
        self._run_prot4()

    def _run_prot4(self, *_):
        Ia_r  = self.p['Ia']
        TDS   = float(self._v_p4_TDS.get())
        Ip_pu = float(self._v_p4_Ip.get())
        If_pu = float(self._v_p4_Ifault.get())
        Ip    = Ip_pu * Ia_r
        If    = If_pu * Ia_r
        CT    = float(self._v_p4_CTr.get())

        I_range = np.linspace(1.01 * Ip, 20 * Ip, 500)
        M = I_range / Ip

        with np.errstate(divide='ignore', invalid='ignore'):
            t_SI = np.where(M > 1.001, TDS * 0.14  / (M**0.02 - 1),  np.inf)
            t_VI = np.where(M > 1.001, TDS * 13.5  / (M       - 1),  np.inf)
            t_EI = np.where(M > 1.001, TDS * 80.0  / (M**2    - 1),  np.inf)

        # Trip times at fault level
        Mf = If / Ip
        def trip_t(formula, Mf):
            if Mf <= 1.001:
                return np.inf
            if formula == 'SI':
                return TDS * 0.14 / (Mf**0.02 - 1)
            elif formula == 'VI':
                return TDS * 13.5 / (Mf - 1)
            else:
                return TDS * 80.0  / (Mf**2 - 1)

        tSI = trip_t('SI', Mf)
        tVI = trip_t('VI', Mf)
        tEI = trip_t('EI', Mf)

        fig = self._t4_fig
        fig.patch.set_facecolor(CARD)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

        ax.semilogy(I_range / Ia_r, np.clip(t_SI, 1e-3, 1e3),
                    color=ACC,  lw=1.8, label='Standard Inverse (SI)')
        ax.semilogy(I_range / Ia_r, np.clip(t_VI, 1e-3, 1e3),
                    color=ACC3, lw=1.8, label='Very Inverse (VI)')
        ax.semilogy(I_range / Ia_r, np.clip(t_EI, 1e-3, 1e3),
                    color=ACC5, lw=1.8, label='Extremely Inverse (EI)')
        ax.axvline(If / Ia_r, color=DIM, lw=1, ls='--',
                   label=f'Fault = {If_pu:.1f}×Ia_r')
        ax.axhline(tSI, color=ACC,  lw=0.8, ls=':', alpha=0.7,
                   label=f'SI trip = {tSI:.3f}s')
        ax.axhline(tVI, color=ACC3, lw=0.8, ls=':', alpha=0.7,
                   label=f'VI trip = {tVI:.3f}s')
        ax.set_xlabel('Current / Ia_rated', color=TXT, fontsize=8)
        ax.set_ylabel('Operating Time [s]', color=TXT, fontsize=8)
        ax.set_title(f'IDMT Protection Curves  |  TDS={TDS}  Ip={Ip_pu:.2f}×Ia_r  CT={CT:.0f}',
                     color=ACC, fontsize=9)
        ax.legend(fontsize=7.5, labelcolor=TXT, facecolor=CARD)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        fig.tight_layout(pad=0.5)
        self._t4_canvas.draw_idle()

        self._t4_info.config(text=(
            f'Pickup current  : {Ip:.1f} A ({Ip_pu:.2f}×In)\n'
            f'Fault current   : {If:.1f} A ({If_pu:.2f}×In)\n'
            f'SI trip time    : {tSI:.4f} s\n'
            f'VI trip time    : {tVI:.4f} s\n'
            f'EI trip time    : {tEI:.4f} s\n'
            f'CT ratio        : {CT:.0f}:1'))
        self._set_status('Protection coordination updated')

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — Speed Controller (PID + Fuzzy)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab5(self):
        left = self._scroll_left(self.t5, width=310)
        right = tk.Frame(self.t5, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._lbl_h(left, '─ Controller Type')
        self._v_t5_type = tk.StringVar(value='PID')
        tk.Radiobutton(left, text='PID Controller',   variable=self._v_t5_type,
                       value='PID', bg=BG, fg=TXT,
                       font=('Consolas', 9), selectcolor=CARD).pack(anchor='w', padx=8)
        tk.Radiobutton(left, text='Fuzzy Controller', variable=self._v_t5_type,
                       value='Fuzzy', bg=BG, fg=TXT,
                       font=('Consolas', 9), selectcolor=CARD).pack(anchor='w', padx=8)

        self._lbl_h(left, '─ PID Gains')
        self._v_Kp = tk.DoubleVar(value=5.0)
        self._v_Ki = tk.DoubleVar(value=10.0)
        self._v_Kd = tk.DoubleVar(value=0.02)
        self._slider(left, 'Kp',  self._v_Kp, 0.1,  80.0, 2)
        self._slider(left, 'Ki',  self._v_Ki, 0.01, 200.0, 2)
        self._slider(left, 'Kd',  self._v_Kd, 0.0,   2.0,  3)

        self._lbl_h(left, '─ Reference & Load')
        self._v_t5_wref  = tk.DoubleVar(value=600.0)
        self._v_t5_tls   = tk.DoubleVar(value=2.0)
        self._v_t5_tload = tk.DoubleVar(value=1500.0)
        self._v_t5_dur   = tk.DoubleVar(value=6.0)
        self._slider(left, 'Speed ref [rpm]',    self._v_t5_wref,  50,  700, 1)
        self._slider(left, 'Load step at [s]',   self._v_t5_tls,   0.5, 10.0, 2)
        self._slider(left, 'Load torque [N·m]',  self._v_t5_tload, 0,   3000, 1)
        self._slider(left, 'Duration [s]',       self._v_t5_dur,   1.0, 20.0, 2)

        self._sep(left)
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill='x', padx=6)
        self._btn(brow, '▶ Run',    self._run_ctrl5)
        self._btn(brow, '■ Stop',   self._stop_ctrl5)

        self._t5_info = tk.Label(left, text='', bg=BG, fg=ACC4,
                                  font=('Consolas', 8), justify='left', wraplength=280)
        self._t5_info.pack(anchor='w', padx=6, pady=6)

        self._t5_fig = Figure(figsize=(9, 6))
        self._t5_canvas = FigureCanvasTkAgg(self._t5_fig, master=right)
        NavigationToolbar2Tk(self._t5_canvas, right).update()
        self._t5_canvas.get_tk_widget().pack(fill='both', expand=True)
        right.bind('<Configure>', lambda e: (self._t5_fig.tight_layout(pad=0.4),
                                              self._t5_canvas.draw_idle()))

    def _stop_ctrl5(self):
        self._stop_flag.set()
        self._set_status('Controller sim stopped')

    def _run_ctrl5(self):
        self._stop_flag.clear()
        self._set_status('Running controller simulation …')
        threading.Thread(target=self._ctrl5_thread, daemon=True).start()

    def _ctrl5_thread(self):
        p      = dict(self.p)
        Ke     = ke_from_p(p)
        Ra     = p['Ra'];  La = p['La'];  J = p['J'];  Bf = p['Bf']
        Vs     = p['Vs'];  Vd = p['Vdrop']
        Ia_max = p['Ia'] * 3.0
        D_max  = 1.0;  D_min = 0.0

        use_fuzzy = (self._v_t5_type.get() == 'Fuzzy')
        Kp = float(self._v_Kp.get())
        Ki = float(self._v_Ki.get())
        Kd = float(self._v_Kd.get())
        w_ref  = float(self._v_t5_wref.get())  * 2.0 * np.pi / 60.0
        t_step = float(self._v_t5_tls.get())
        Tl     = float(self._v_t5_tload.get())
        dur    = float(self._v_t5_dur.get())

        dt      = 1e-4      # controller time step
        n_steps = int(dur / dt) + 1
        t_arr   = np.zeros(n_steps)
        ia_arr  = np.zeros(n_steps)
        w_arr   = np.zeros(n_steps)
        D_arr   = np.zeros(n_steps)

        ia  = 0.0;  omega = 0.0
        integ = 0.0;  prev_err = 0.0
        D_fuzzy = 0.3
        e_max  = w_ref if w_ref > 1e-6 else 1.0
        de_max = e_max * 2.0

        for k in range(n_steps):
            if self._stop_flag.is_set():
                break
            t = k * dt
            err = w_ref - omega

            if use_fuzzy:
                derr = (err - prev_err) / dt
                D_fuzzy += _fuzzy_delta_D(
                    float(np.clip(err  / e_max,  -1, 1)),
                    float(np.clip(derr / de_max, -1, 1))
                ) * 0.05
                D_cmd = float(np.clip(D_fuzzy, D_min, D_max))
            else:
                integ += err * dt
                derr   = (err - prev_err) / dt
                u      = Kp * err + Ki * integ + Kd * derr
                D_cmd  = float(np.clip(u, D_min, D_max))
            prev_err = err

            # Converter average voltage (unipolar 2-quadrant)
            V_on  = Vs - 2.0 * Vd
            V_off = -2.0 * Vd
            va    = D_cmd * V_on + (1.0 - D_cmd) * V_off

            Tl_now = Tl if t >= t_step else 0.0
            dia    = (va - Ra * ia - Ke * omega) / La
            dw     = (Ke * ia - Tl_now - Bf * omega) / J
            ia     = float(np.clip(ia + dia * dt, -Ia_max, Ia_max))
            omega  = float(max(omega + dw * dt, -0.1))

            t_arr[k]  = t
            ia_arr[k] = ia
            w_arr[k]  = omega * 60.0 / (2.0 * np.pi)
            D_arr[k]  = D_cmd

        t_arr   = t_arr[:k+1]
        ia_arr  = ia_arr[:k+1]
        w_arr   = w_arr[:k+1]
        D_arr   = D_arr[:k+1]
        w_ref_rpm = w_ref * 60.0 / (2.0 * np.pi)

        # Performance metrics
        ss_err = abs(w_arr[-1] - w_ref_rpm) if len(w_arr) > 0 else 0
        overshoot = max(0, (np.max(w_arr) - w_ref_rpm)) if len(w_arr) > 0 else 0

        self.after(0, lambda: self._update_ctrl5(
            t_arr, ia_arr, w_arr, D_arr, w_ref_rpm,
            t_step, ss_err, overshoot, use_fuzzy))

    def _update_ctrl5(self, t, ia, rpm, D, ref, t_step, ss_err, overshoot, fuzzy):
        fig = self._t5_fig
        fig.patch.set_facecolor(CARD)
        fig.clf()
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

        ctrl_name = 'Fuzzy' if fuzzy else 'PID'
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        ax1.plot(t, rpm, color=ACC,  lw=1.2, label='ω actual')
        ax1.axhline(ref,    color=ACC5, lw=1, ls='--', label=f'ω_ref={ref:.0f}rpm')
        ax1.axvline(t_step, color=DIM, lw=0.8, ls=':', label='Load step')
        ax1.set_ylabel('Speed [rpm]', color=TXT, fontsize=8)
        ax1.yaxis.label.set_color(TXT)
        ax1.set_title(f'{ctrl_name} Speed Controller — Step Load Response', color=ACC, fontsize=9)
        ax1.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)

        ax2.plot(t, ia, color=ACC3, lw=1.0, label='ia(t) [A]')
        ax2.axhline( self.p['Ia'],  color=ACC5, lw=0.8, ls='--')
        ax2.axhline(-self.p['Ia'],  color=ACC5, lw=0.8, ls='--')
        ax2.set_ylabel('Current [A]', color=TXT, fontsize=8)
        ax2.yaxis.label.set_color(TXT)
        ax2.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)

        ax3.plot(t, D * 100, color=ACC4, lw=1.0, label='Duty cycle [%]')
        ax3.set_ylabel('D [%]', color=TXT, fontsize=8)
        ax3.set_xlabel('Time [s]', color=TXT, fontsize=8)
        ax3.yaxis.label.set_color(TXT)
        ax3.xaxis.label.set_color(TXT)
        ax3.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)

        fig.tight_layout(pad=0.5)
        self._t5_canvas.draw_idle()

        self._t5_info.config(text=(
            f'Controller     : {ctrl_name}\n'
            f'Final speed    : {rpm[-1]:.1f} rpm\n'
            f'Reference      : {ref:.1f} rpm\n'
            f'SS error       : {ss_err:.2f} rpm\n'
            f'Overshoot      : {overshoot:.2f} rpm\n'
            f'Load step torque applied at {t_step:.1f} s'))
        self._set_status(f'{ctrl_name} done  |  SS_err={ss_err:.2f}rpm  OS={overshoot:.2f}rpm')

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6 — Thermal & Economic Analysis
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab6(self):
        # Two sub-frames side by side
        lp = tk.Frame(self.t6, bg=BG)
        lp.pack(side='left', fill='both', expand=True, padx=2)
        rp = tk.Frame(self.t6, bg=BG)
        rp.pack(side='left', fill='both', expand=True, padx=2)

        # ── Thermal ──
        tk.Label(lp, text='THERMAL ANALYSIS', bg=BG, fg=ACC,
                 font=('Consolas', 10, 'bold')).pack(anchor='w', padx=6, pady=(6, 2))

        ctrl_th = tk.Frame(lp, bg=BG)
        ctrl_th.pack(fill='x')
        self._v_Rth  = tk.DoubleVar(value=0.015)
        self._v_Cth  = tk.DoubleVar(value=10000.0)
        self._v_Tamb = tk.DoubleVar(value=25.0)
        self._v_t6_Ia = tk.DoubleVar(value=float(DEF['Ia']))
        self._slider(ctrl_th, 'Rth [°C/W]',   self._v_Rth,   0.001, 0.1,   4, self._run_thermal6)
        self._slider(ctrl_th, 'Cth [J/°C]',   self._v_Cth,   500,   50000, 1, self._run_thermal6)
        self._slider(ctrl_th, 'T_amb [°C]',   self._v_Tamb,  0,     60,    1, self._run_thermal6)
        self._slider(ctrl_th, 'Load Ia [A]',  self._v_t6_Ia, 0,     700,   1, self._run_thermal6)

        tk.Frame(ctrl_th, bg=BORDER, height=1).pack(fill='x', padx=6, pady=2)
        brow = tk.Frame(ctrl_th, bg=BG)
        brow.pack(fill='x', padx=6)
        self._btn(brow, '▶ Run Thermal', self._run_thermal6)

        self._t6_th_info = tk.Label(ctrl_th, text='', bg=BG, fg=ACC4,
                                     font=('Consolas', 8), justify='left')
        self._t6_th_info.pack(anchor='w', padx=6, pady=4)

        self._t6_fig_th = Figure(figsize=(5, 3.5))
        cvs_th = FigureCanvasTkAgg(self._t6_fig_th, master=lp)
        NavigationToolbar2Tk(cvs_th, lp).update()
        cvs_th.get_tk_widget().pack(fill='both', expand=True)
        lp.bind('<Configure>', lambda e: (self._t6_fig_th.tight_layout(pad=0.4),
                                          cvs_th.draw_idle()))
        self._t6_cvs_th = cvs_th

        # ── Economic ──
        tk.Label(rp, text='ECONOMIC ANALYSIS', bg=BG, fg=ACC3,
                 font=('Consolas', 10, 'bold')).pack(anchor='w', padx=6, pady=(6, 2))

        ctrl_ec = tk.Frame(rp, bg=BG)
        ctrl_ec.pack(fill='x')
        self._v_Ec_cost  = tk.DoubleVar(value=0.12)
        self._v_Ec_hpd   = tk.DoubleVar(value=16.0)
        self._v_Ec_dpy   = tk.DoubleVar(value=300.0)
        self._slider(ctrl_ec, 'Energy [$/kWh]', self._v_Ec_cost, 0.03, 0.5,  3, self._run_economic6)
        self._slider(ctrl_ec, 'Hours/day',      self._v_Ec_hpd,  1,   24,    1, self._run_economic6)
        self._slider(ctrl_ec, 'Days/year',      self._v_Ec_dpy,  50,  365,   1, self._run_economic6)

        tk.Frame(ctrl_ec, bg=BORDER, height=1).pack(fill='x', padx=6, pady=2)
        brow2 = tk.Frame(ctrl_ec, bg=BG)
        brow2.pack(fill='x', padx=6)
        self._btn(brow2, '▶ Run Economic', self._run_economic6)

        self._t6_ec_info = tk.Label(ctrl_ec, text='', bg=BG, fg=ACC3,
                                     font=('Consolas', 8), justify='left')
        self._t6_ec_info.pack(anchor='w', padx=6, pady=4)

        self._t6_fig_ec = Figure(figsize=(5, 3.5))
        cvs_ec = FigureCanvasTkAgg(self._t6_fig_ec, master=rp)
        NavigationToolbar2Tk(cvs_ec, rp).update()
        cvs_ec.get_tk_widget().pack(fill='both', expand=True)
        rp.bind('<Configure>', lambda e: (self._t6_fig_ec.tight_layout(pad=0.4),
                                          cvs_ec.draw_idle()))
        self._t6_cvs_ec = cvs_ec

        self._run_thermal6()
        self._run_economic6()

    def _run_thermal6(self, *_):
        p    = dict(self.p)
        Ia   = float(self._v_t6_Ia.get())
        Rth  = float(self._v_Rth.get())
        Cth  = float(self._v_Cth.get())
        Tamb = float(self._v_Tamb.get())
        Ra   = p['Ra'];  Bf = p['Bf']
        omega_r = p['n_rated'] * 2.0 * np.pi / 60.0

        P_cu  = Ra * Ia ** 2
        P_fric = Bf * omega_r ** 2
        P_iron = 0.01 * p['P_hp'] * 745.7
        P_loss = P_cu + P_fric + P_iron
        T_ss   = Tamb + P_loss * Rth
        tau_th = Rth * Cth

        t_th   = np.linspace(0, min(5 * tau_th, 7200), 1000)
        T_th   = Tamb + P_loss * Rth * (1 - np.exp(-t_th / tau_th))

        fig = self._t6_fig_th
        fig.patch.set_facecolor(CARD)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

        ax.plot(t_th / 60, T_th, color=ACC3, lw=1.8, label='Winding Temp')
        ax.axhline(130, color=ACC5, lw=1, ls='--', label='Class B limit 130°C')
        ax.axhline(155, color=ACC2, lw=1, ls='--', label='Class F limit 155°C')
        ax.axhline(T_ss, color=ACC4, lw=0.8, ls=':', label=f'T_ss={T_ss:.1f}°C')
        ax.set_xlabel('Time [min]', color=TXT, fontsize=8)
        ax.set_ylabel('Temperature [°C]', color=TXT, fontsize=8)
        ax.set_title('Thermal Model — Winding Temperature Rise', color=ACC, fontsize=9)
        ax.legend(fontsize=7, labelcolor=TXT, facecolor=CARD)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        fig.tight_layout(pad=0.5)
        self._t6_cvs_th.draw_idle()

        self._t6_th_info.config(text=(
            f'P_copper  = {P_cu/1000:.2f} kW\n'
            f'P_friction= {P_fric/1000:.2f} kW\n'
            f'P_iron    = {P_iron/1000:.2f} kW\n'
            f'P_total   = {P_loss/1000:.2f} kW\n'
            f'T_ambient = {Tamb:.1f} °C\n'
            f'T_steady  = {T_ss:.1f} °C\n'
            f'τ_thermal = {tau_th/60:.1f} min'))

    def _run_economic6(self, *_):
        p    = dict(self.p)
        Ra   = p['Ra'];  Bf = p['Bf']
        Ia   = float(self._v_t6_Ia.get())
        cost = float(self._v_Ec_cost.get())
        hpd  = float(self._v_Ec_hpd.get())
        dpy  = float(self._v_Ec_dpy.get())
        omega_r = p['n_rated'] * 2.0 * np.pi / 60.0

        P_cu   = Ra * Ia ** 2
        P_fric = Bf * omega_r ** 2
        P_iron = 0.01 * p['P_hp'] * 745.7
        P_total = P_cu + P_fric + P_iron

        h_yr    = hpd * dpy
        E_yr    = P_total / 1000.0 * h_yr   # kWh/year
        cost_yr = E_yr * cost
        co2_yr  = E_yr * 0.5 / 1000.0       # tonnes CO2 (0.5 kg/kWh)

        # Payback for 2% efficiency improvement motor
        P_rated = p['P_hp'] * 745.7
        eta_now  = P_rated / (P_rated + P_total)
        eta_new  = eta_now + 0.02
        E_saved  = (P_total - P_total * (1 - eta_now) / (1 - eta_new)) / 1000 * h_yr
        savings  = E_saved * cost
        capital  = 5000.0
        payback  = capital / savings if savings > 0 else float('inf')

        cats   = ['Copper\nLoss', 'Friction\nLoss', 'Iron\nLoss', 'Total\nLoss']
        values = [P_cu/1000, P_fric/1000, P_iron/1000, P_total/1000]
        colors_b = [ACC, ACC2, ACC3, ACC5]

        fig = self._t6_fig_ec
        fig.patch.set_facecolor(CARD)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

        bars = ax.bar(cats, values, color=colors_b, edgecolor=BORDER, linewidth=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005 * max(values),
                    f'{v:.2f} kW', ha='center', color=TXT, fontsize=7)
        ax.set_ylabel('Power Loss [kW]', color=TXT, fontsize=8)
        ax.set_title('Power Loss Breakdown', color=ACC3, fontsize=9)
        ax.yaxis.label.set_color(TXT)
        fig.tight_layout(pad=0.5)
        self._t6_cvs_ec.draw_idle()

        self._t6_ec_info.config(text=(
            f'Energy/year = {E_yr:.0f} kWh\n'
            f'Cost/year   = ${cost_yr:,.0f}\n'
            f'CO₂/year    = {co2_yr:.2f} tonnes\n'
            f'Efficiency  = {eta_now*100:.2f}%\n'
            f'2% η gain saves ${savings:,.0f}/yr\n'
            f'Payback ($5k upgrade) = {payback:.1f} yr'))

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 7 — Harmonic & Power Quality
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab7(self):
        left = self._scroll_left(self.t7, width=300)
        right = tk.Frame(self.t7, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._lbl_h(left, '─ Harmonic Analysis')
        self._v_t7_load = tk.DoubleVar(value=1.0)
        self._v_t7_cyc  = tk.DoubleVar(value=20)
        self._slider(left, 'Load level [pu]',    self._v_t7_load, 0.1, 1.0, 2)
        self._slider(left, 'Switching cycles',   self._v_t7_cyc,  5,   100, 1)

        self._sep(left)
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill='x', padx=6)
        self._btn(brow, '▶ Analyze', self._run_harmonics7)

        self._t7_info = tk.Label(left, text='', bg=BG, fg=ACC4,
                                  font=('Consolas', 8), justify='left', wraplength=280)
        self._t7_info.pack(anchor='w', padx=6, pady=6)

        tk.Label(left, text=(
            'IEEE 519 Limits (ISC/IL < 20):\n'
            '  h < 11:  4.0 %\n'
            '  h 11-16: 2.0 %\n'
            '  h 17-22: 1.5 %\n'
            '  h > 22:  0.6 %\n'
            '  THD:     5.0 %'),
            bg=BG, fg=DIM, font=('Consolas', 8),
            justify='left').pack(anchor='w', padx=6)

        self._t7_fig = Figure(figsize=(9, 6))
        self._t7_canvas = FigureCanvasTkAgg(self._t7_fig, master=right)
        NavigationToolbar2Tk(self._t7_canvas, right).update()
        self._t7_canvas.get_tk_widget().pack(fill='both', expand=True)
        right.bind('<Configure>', lambda e: (self._t7_fig.tight_layout(pad=0.4),
                                              self._t7_canvas.draw_idle()))

    def _run_harmonics7(self, *_):
        p    = dict(self.p)
        load = float(self._v_t7_load.get())
        n_cyc = int(float(self._v_t7_cyc.get()))

        D, ripple, V_on, V_off = duty_ripple(p, drop=True, bipolar=False)
        Ia_ss  = load * p['Ia']
        f_sw   = p['f_sw']
        T_sw   = 1.0 / f_sw
        Ke     = ke_from_p(p)
        Ra     = p['Ra'];  La = p['La']
        omega_ss = (p['Vn'] - Ia_ss * Ra) / Ke

        # Simulate steady-state current waveform
        n_pts  = 200 * n_cyc
        t_ss   = np.linspace(0, n_cyc * T_sw, n_pts, endpoint=False)
        dt_ss  = t_ss[1] - t_ss[0]

        ia = np.zeros(n_pts)
        ia[0] = Ia_ss
        for k in range(1, n_pts):
            phase = (t_ss[k] % T_sw) / T_sw
            va    = V_on if phase < D else V_off
            dia   = (va - Ra * ia[k-1] - Ke * omega_ss) / La
            ia[k] = ia[k-1] + dia * dt_ss

        # FFT
        N      = len(ia)
        window = np.hanning(N)
        spectrum = np.abs(np.fft.rfft(ia * window)) * 2.0 / N
        freqs    = np.fft.rfftfreq(N, d=dt_ss)

        # Find fundamental (DC motor: fundamental ≈ DC + ripple at f_sw)
        I_dc   = np.mean(ia)
        I_rms  = float(np.sqrt(np.mean(ia**2)))
        I_peak = float(np.max(np.abs(ia)))
        # Harmonics of switching frequency
        harm_ords  = np.arange(1, 21)
        harm_freqs = harm_ords * f_sw
        harm_amps  = []
        for hf in harm_freqs:
            idx = np.argmin(np.abs(freqs - hf))
            harm_amps.append(float(spectrum[idx]))
        harm_amps = np.array(harm_amps)
        I1 = harm_amps[0] if harm_amps[0] > 1e-3 else I_rms
        THD = float(np.sqrt(np.sum(harm_amps[1:]**2)) / (I1 + 1e-12) * 100)
        crest = I_peak / (I_rms + 1e-12)
        cumTHD = np.array([float(np.sqrt(np.sum(harm_amps[1:k+1]**2)) / (I1 + 1e-12) * 100)
                           for k in range(1, len(harm_amps))])

        fig = self._t7_fig
        fig.patch.set_facecolor(CARD)
        fig.clf()
        axes = fig.subplots(2, 2)
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        # Time domain
        cycles_show = min(3, n_cyc)
        mask = t_ss < cycles_show * T_sw
        ax1.plot(t_ss[mask] * 1000, ia[mask], color=ACC, lw=1.0)
        ax1.set_xlabel('Time [ms]', color=TXT, fontsize=7)
        ax1.set_ylabel('ia [A]',   color=TXT, fontsize=7)
        ax1.set_title('Armature Current (time domain)', color=ACC, fontsize=8)
        ax1.xaxis.label.set_color(TXT);  ax1.yaxis.label.set_color(TXT)

        # Frequency spectrum
        show_n = min(100, len(freqs))
        ax2.bar(freqs[:show_n] / 1000, spectrum[:show_n],
                width=freqs[1]/1000 if len(freqs)>1 else 0.1,
                color=ACC4, edgecolor=BG2, linewidth=0)
        ax2.set_xlabel('Frequency [kHz]', color=TXT, fontsize=7)
        ax2.set_ylabel('Amplitude [A]',  color=TXT, fontsize=7)
        ax2.set_title('FFT Spectrum', color=ACC, fontsize=8)
        ax2.xaxis.label.set_color(TXT);  ax2.yaxis.label.set_color(TXT)

        # Harmonic bar chart
        ax3.bar(harm_ords, harm_amps / (I1 + 1e-12) * 100,
                color=ACC3, edgecolor=BG2, linewidth=0)
        ax3.axhline(5.0, color=ACC5, lw=1, ls='--', label='IEEE519 THD 5%')
        ax3.set_xlabel('Harmonic order (× f_sw)', color=TXT, fontsize=7)
        ax3.set_ylabel('Amplitude [% of I1]', color=TXT, fontsize=7)
        ax3.set_title('Harmonic Spectrum [%]', color=ACC, fontsize=8)
        ax3.legend(fontsize=6.5, labelcolor=TXT, facecolor=CARD)
        ax3.xaxis.label.set_color(TXT);  ax3.yaxis.label.set_color(TXT)

        # Cumulative THD
        ax4.plot(np.arange(1, len(cumTHD)+1), cumTHD, color=ACC2, lw=1.5, marker='o', ms=3)
        ax4.axhline(5.0, color=ACC5, lw=1, ls='--', label='IEEE519 5%')
        ax4.set_xlabel('Harmonics included', color=TXT, fontsize=7)
        ax4.set_ylabel('Cumulative THD [%]', color=TXT, fontsize=7)
        ax4.set_title('Cumulative THD', color=ACC, fontsize=8)
        ax4.legend(fontsize=6.5, labelcolor=TXT, facecolor=CARD)
        ax4.xaxis.label.set_color(TXT);  ax4.yaxis.label.set_color(TXT)

        fig.tight_layout(pad=0.5)
        self._t7_canvas.draw_idle()

        self._t7_info.config(text=(
            f'I_dc   = {I_dc:.1f} A\n'
            f'I_rms  = {I_rms:.1f} A\n'
            f'I_peak = {I_peak:.1f} A\n'
            f'THD    = {THD:.2f} %\n'
            f'Crest  = {crest:.3f}\n'
            f'I1(f_sw)={harm_amps[0]:.2f} A'))
        self._set_status(f'Harmonic analysis done  |  THD={THD:.2f}%')

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 8 — Comprehensive Analysis
    # ═════════════════════════════════════════════════════════════════════════

    def _build_tab8(self):
        top = tk.Frame(self.t8, bg=BG)
        top.pack(fill='x', padx=6, pady=4)
        self._btn(top, '▶ Refresh All', self._refresh_all8)

        # Summary text
        self._t8_text = scrolledtext.ScrolledText(
            self.t8, bg=CARD, fg=TXT, font=('Consolas', 8),
            height=9, relief='flat', wrap='none')
        self._t8_text.pack(fill='x', padx=4, pady=(0, 2))

        # Figure
        fig_frame = tk.Frame(self.t8, bg=BG)
        fig_frame.pack(fill='both', expand=True, padx=4, pady=2)
        self._t8_fig = Figure(figsize=(12, 5))
        self._t8_canvas = FigureCanvasTkAgg(self._t8_fig, master=fig_frame)
        NavigationToolbar2Tk(self._t8_canvas, fig_frame).update()
        self._t8_canvas.get_tk_widget().pack(fill='both', expand=True)
        fig_frame.bind('<Configure>', lambda e: (self._t8_fig.tight_layout(pad=0.4),
                                                  self._t8_canvas.draw_idle()))
        self._refresh_all8()

    def _refresh_all8(self, *_):
        p   = dict(self.p)
        Ke  = ke_from_p(p)
        Tr  = rated_torque(p)
        P_W = p['P_hp'] * 745.7
        omega_r = p['n_rated'] * 2.0 * np.pi / 60.0

        Da, Ra_pp, _, _ = duty_ripple(p, drop=False, bipolar=False)
        Dc, Rc_pp, _, _ = duty_ripple(p, drop=True,  bipolar=False)
        Dd, Rd_pp, _, _ = duty_ripple(p, drop=True,  bipolar=True)

        P_cu   = p['Ra'] * p['Ia'] ** 2
        P_fric = p['Bf'] * omega_r ** 2
        P_iron = 0.01 * P_W
        P_loss = P_cu + P_fric + P_iron
        eta    = P_W / (P_W + P_loss) * 100

        lines = [
            '═'*100,
            '  COMPREHENSIVE ANALYSIS — 200 hp  250 V  600 rpm DC Motor — 4-Quadrant GTO Drive',
            '═'*100,
            f'  Ke=Kt={Ke:.4f} V·s/rad   Tr={Tr:.1f} N·m   ηrated={eta:.2f}%   P_loss={P_loss/1000:.2f} kW',
            f'  (a) Ideal 2-quad D={Da*100:.4f}%  ΔI={Ra_pp:.1f}A  |  '
            f'(c) Drops D={Dc*100:.4f}%  ΔI={Rc_pp:.1f}A  |  '
            f'(d) Bipolar D={Dd*100:.4f}%  ΔI={Rd_pp:.0f}A  ({Rd_pp/max(Rc_pp,1e-9):.1f}× worse)',
            f'  τe={p["La"]/p["Ra"]*1000:.2f}ms   J={p["J"]:.1f}kg·m²   Bf={p["Bf"]:.3f}N·m·s',
            '═'*100,
        ]
        self._t8_text.config(state='normal')
        self._t8_text.delete('1.0', 'end')
        self._t8_text.insert('end', '\n'.join(lines))
        self._t8_text.config(state='disabled')

        # 4-subplot comprehensive figure
        fig = self._t8_fig
        fig.patch.set_facecolor(CARD)
        fig.clf()
        axes = fig.subplots(1, 4)
        ax1, ax2, ax3, ax4 = axes
        for ax in axes:
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        # ax1: Torque-speed (4 quadrants)
        omega_range = np.linspace(-omega_r * 1.1, omega_r * 1.1, 400)
        rpm_range   = omega_range * 60.0 / (2.0 * np.pi)
        T_motoring  = Ke * p['Ia'] - p['Bf'] * omega_range
        T_braking   = Ke * (-p['Ia']) - p['Bf'] * omega_range

        ax1.axhline(0, color=BORDER, lw=0.8)
        ax1.axvline(0, color=BORDER, lw=0.8)
        ax1.plot(rpm_range, T_motoring, color=ACC4, lw=1.8, label='Motoring (Ia>0)')
        ax1.plot(rpm_range, T_braking,  color=ACC5, lw=1.8, label='Braking (Ia<0)')
        ax1.fill_betweenx([0, Tr], [0, 0], [p['n_rated'], p['n_rated']],
                          alpha=0.12, color=ACC4)
        ax1.text(p['n_rated']*0.5, Tr*0.5, 'Q1', color=ACC4, fontsize=8, ha='center')
        ax1.text(-p['n_rated']*0.5, Tr*0.5, 'Q2', color=ACC3, fontsize=8, ha='center')
        ax1.text(-p['n_rated']*0.5, -Tr*0.5, 'Q3', color=DIM, fontsize=8, ha='center')
        ax1.text(p['n_rated']*0.5, -Tr*0.5, 'Q4', color=ACC2, fontsize=8, ha='center')
        ax1.set_xlabel('Speed [rpm]', color=TXT, fontsize=7)
        ax1.set_ylabel('Torque [N·m]', color=TXT, fontsize=7)
        ax1.set_title('Torque-Speed 4Q', color=ACC, fontsize=8)
        ax1.legend(fontsize=6, labelcolor=TXT, facecolor=CARD)
        ax1.xaxis.label.set_color(TXT);  ax1.yaxis.label.set_color(TXT)

        # ax2: Efficiency map
        Ia_grid = np.linspace(0.1 * p['Ia'], 1.2 * p['Ia'], 50)
        om_grid = np.linspace(0.1 * omega_r, 1.1 * omega_r, 50)
        IA, OM  = np.meshgrid(Ia_grid, om_grid)
        P_out   = Ke * IA * OM
        P_in_g  = P_out + p['Ra'] * IA**2 + p['Bf'] * OM**2
        ETA     = np.clip(P_out / np.where(P_in_g > 1, P_in_g, 1) * 100, 0, 100)
        cs = ax2.contourf(IA, OM * 60 / (2*np.pi), ETA,
                          levels=np.arange(60, 101, 2), cmap='plasma')
        fig.colorbar(cs, ax=ax2, label='η [%]')
        ax2.set_xlabel('Ia [A]', color=TXT, fontsize=7)
        ax2.set_ylabel('Speed [rpm]', color=TXT, fontsize=7)
        ax2.set_title('Efficiency Map', color=ACC, fontsize=8)
        ax2.xaxis.label.set_color(TXT);  ax2.yaxis.label.set_color(TXT)

        # ax3: Power flow bar chart
        P_elec_in = P_W + P_loss
        items     = ['P_electrical\nin', 'P_copper\nloss', 'P_friction\nloss',
                     'P_iron\nloss', 'P_shaft\nout']
        values_p  = [P_elec_in/1000, P_cu/1000, P_fric/1000, P_iron/1000, P_W/1000]
        col_p     = [ACC, ACC5, ACC3, ACC2, ACC4]
        ax3.barh(items, values_p, color=col_p, edgecolor=BORDER, linewidth=0.8)
        for i, v in enumerate(values_p):
            ax3.text(v + 0.005 * max(values_p), i, f'{v:.1f} kW',
                     va='center', color=TXT, fontsize=7)
        ax3.set_xlabel('Power [kW]', color=TXT, fontsize=7)
        ax3.set_title('Power Flow', color=ACC, fontsize=8)
        ax3.xaxis.label.set_color(TXT)

        # ax4: Summary metrics radar
        categories = ['D_ideal\n×100', 'D_drops\n×100', 'η%/100',
                      'ΔI_ideal\n/Ia', 'ΔI_drops\n/Ia']
        vals = [Da*100*5, Dc*100*5, eta/100*100,
                Ra_pp / p['Ia'] * 100, Rc_pp / p['Ia'] * 100]
        # Normalise to 0-100 for radar
        v_norm = np.clip(vals, 0, 100)
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        v_plot = v_norm.copy()

        ax4.set_title('Metric Summary', color=ACC, fontsize=8)
        x_pos = np.arange(len(categories))
        ax4.bar(x_pos, v_norm, color=[ACC, ACC3, ACC4, ACC5, ACC2],
                edgecolor=BORDER, linewidth=0.8)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, color=TXT, fontsize=6.5)
        ax4.set_ylabel('Normalised value', color=TXT, fontsize=7)
        ax4.yaxis.label.set_color(TXT)

        fig.tight_layout(pad=0.5)
        self._t8_canvas.draw_idle()
        self._set_status('Comprehensive analysis refreshed')


# ─────────────────────────────────────────────────────────────────────────────
# Problem text constant
# ─────────────────────────────────────────────────────────────────────────────

PROBLEM_TEXT = """\
PROBLEM — 200 hp DC Motor, 4-Quadrant GTO Converter
====================================================

A 200 hp, 250 V, 600 r/min DC motor is driven by a 4-quadrant converter.
GTOs are used, operating at a switching frequency of 125 Hz.
The voltage of the DC source is 280 V.

MOTOR CHARACTERISTICS
─────────────────────
  Armature resistance  Ra  = 12 mΩ  (0.012 Ω)
  Armature inductance  La  = 350 µH (3.50 × 10⁻⁴ H)
  Rated armature current   = 620 A
  Rated speed              = 600 rpm
  Rated power              = 200 hp = 149.2 kW

STARTUP OPERATION
─────────────────
  • Average armature current maintained at 620 A
  • Q4 GTO: always closed
  • Q3 GTO: always open
  → Converter acts as a 2-QUADRANT converter during startup

QUESTIONS
─────────────────────────────────────────────────────────
  (a) Assuming negligible switch voltage drops, calculate
      the duty cycle needed to establish an average current
      of 620 A in the armature when it is STALLED.

  (b) Calculate the peak-to-peak current ripple under
      these (ideal, stalled) conditions.

  (c) If the voltage drop across the switches is 2 V,
      calculate the new duty cycle and peak-to-peak ripple.

  (d) During the start-up phase, would the current ripple
      be seriously affected if the converter were operated
      as a 4-quadrant unit?

KEY FORMULAS
─────────────────────────────────────────────────────────
  2-quadrant unipolar mode (startup):
    V_on  = Vs − 2·Vdrop       (Q1 + Q4 conducting)
    V_off = −2·Vdrop            (freewheeling D2 + Q4)
    D·V_on + (1−D)·V_off = Ia·Ra   (stalled: E=0)
    → D = (Ia·Ra − V_off) / (V_on − V_off)
    → ΔI = (V_on − V_off) · D · (1−D) / (La · f_sw)

  4-quadrant bipolar mode:
    V_on  = +(Vs − 2·Vdrop)
    V_off = −(Vs − 2·Vdrop)
    → D ≈ 0.513  (nearly 50%)
    → Much larger voltage swing → ~19× larger ripple

  Back-EMF constant:
    Ke = (Vn − Ia·Ra) / ωr = (250 − 620×0.012) / (600×2π/60)
       ≈ 3.861 V·s/rad

  Electrical time constant:  τe = La / Ra = 29.2 ms
  Switching period:          T  = 1/125 = 8.0 ms

USE THE ⚙ PARAMETERS TAB FOR INTERACTIVE CALCULATIONS
USE THE 📈 SIMULATION TAB FOR ODE TRANSIENT SIMULATION
"""


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = App()
    app.mainloop()
