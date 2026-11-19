"""model.py -- a growing Markov-blanket organ (Palacios et al. 2020, every agent one ADEM step) that keeps
two bits and a position in its tissues' clocks through cuts, and lets the two bits take turns.  See README.md.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, LinAlgError

from spm import spm_speye, spm_DEM_R, spm_DEM_embed, spm_diff_vec, spm_logdet, spm_inv, spm_dx, to_csr

# ---- the paper's constants ---------------------------------------------------------------
E_d, E_n, E_s, E_dt = 1, 2, 1, 2        # DEM orders, smoothness, dt
N_TYPE = np.array([1, 6, 9])            # form A: internal, active, sensory
R_RAD = np.array([0.0, 1.0, 3.0])       # template radii
K_DECAY = 2.0                           # field kernel exp(-2 d^2)
NG = 16
# ---- the organ's -------------------------------------------------------------------------
Q = 16.0                                # cells per blanket; ensembles per body
CELL_SCALE = 0.05                       # blanket radius in the organ
EMIT_FLOOR = 0.25                       # loudness floor (head count)
THETA = 0.5 / Q                         # deficit that counts
TAU, REFRACTORY, SURPLUS = 3, 4, 0.5    # steps short to divide; between; surplus
V_BIAS = 2.0                            # a determined belief
ORGAN_BIAS = 0.5                        # seeding belief
POS_BIAS, POS_NOISE, SPREAD = 0.5, 0.125, 0.5   # seeding positions
PRIOR_PREC, PRIOR_FLOOR = np.exp(4.0), np.exp(-4.0)   # core prior; floor
ORGAN_FATE_PREC = np.exp(1.0)           # daughter ensemble's fate prior
ORGAN_FP_ACT = 0.11                     # active ensemble's expected active signal
BIRTH_R = 1.3                           # birth radius from the core
PULL_R = (1.0, 2.0)                     # active, sensory radii (form A)
PULL = 0.05                             # pull per step
OFFSET, OFFSET_SKIN = 0.3, 1.2          # birth offset off the ring
# ---- the forms ---------------------------------------------------------------------------
FORM_B = np.array([1, 3, 12])           # form B
PULL_R_B = (1.0, 2.5)                   # form B's radii
FORM_DECAY, FORM_KAPPA = 0.02, 0.1      # belief decay; evidence weight
# ---- the clocks, the bits, the walk and the repair ---------------------------------------
T_RING = (10.0, 16.18)                  # inner, outer (and core) periods
T_SPREAD, AMP = 0.05, 0.3               # period spread; gain 1 + AMP sin(phase)
RING_SPLIT = 1.5                        # inner ring radius
SIGMA, REPEL = 1.0, 0.3                 # link exp(-d^2 / 2 SIGMA^2) - REPEL
WRITE = np.array([-1.0, 1.0])           # windings written: inner B, outer A
FEEDBACK, READ, READ_TAU = 0.4, 0.5, 10.0   # form's tilt on its winding; read weight, time
QUIET, QUIET_UNTIL, READ_MIN = 30, 0.5, 3   # new tissue's quiet steps, read; fewest to read a ring
WALK_SPEED, WALK_TAU = 0.02, 20.0       # walk: cycles per step, correlation time
COPY_GAIN, HOLD_PULL = 0.1, 0.3         # copy's lock gain; its hold on a regrowing ring
STABLE, ADAPT = 30, 50.0                # steps to count stable; rate drift time
# ---- turn-taking -------------------------------------------------------------------------
TURN_EPS, TURN_THRESH, TURN_NOISE = 0.15, 8.0, 0.3   # pattern depth; threshold; noise
AT = dict(clocks=150, write=180, vote=250, turns=300, walk=380)   # step each part starts
NAMES = ["internal", "active", "sensory"]
KEYS = ("l", "g", "c", "s", "b")        # cell kinds, ensemble kinds, views, pictures, body
PER_CELL = ("l", "c", "s")
cw = lambda x: np.angle(np.exp(1j * x))  # wrap to (-pi, pi]


def softmax(v):
    e = np.exp(v - v.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def foxhound(x, y, k=K_DECAY):
    """Field sensed at each column of x from the secretions y of the others: (m, n)."""
    d = ((x[:, :, None] - x[:, None, :]) ** 2).sum(axis=0)
    W = np.exp(-k * d)
    np.fill_diagonal(W, 0.0)
    return np.ascontiguousarray(y) @ W.T       # fixed layout: the same sum order every run


def picture(vc):
    """The picture of a view: exp(v.s) = softmax(v.c), volume 1."""
    return vc - np.log(np.exp(vc).sum(axis=0, keepdims=True))


def build_priors():
    """The paper's template: positions, one-hot kinds and the fingerprint P_sen[signal, kind]."""
    pos, typ = [], []
    for j in range(3):
        for i in range(1, N_TYPE[j] + 1):
            th = 2 * np.pi * i / N_TYPE[j]
            pos.append(R_RAD[j] * np.array([np.sin(th), np.cos(th)]))
            typ.append(j)
    pos = np.array(pos).T
    typ = np.array(typ)
    T = np.zeros((3, N_TYPE.sum()))
    T[typ, np.arange(N_TYPE.sum())] = 1.0
    sen = np.zeros_like(T)                    # cell by cell, as the original
    for i in range(T.shape[1]):
        for j in range(T.shape[1]):
            if i != j:
                dd = pos[:, i] - pos[:, j]
                sen[:, i] += np.exp(-K_DECAY * (dd @ dd)) * T[:, j]
    P = np.zeros((3, 3))
    for i in range(3):
        P[:, i] = sen[:, typ == i].mean(axis=1)
    return pos, T, P


def form_fp(n_type):
    """The fingerprint of a blanket of n_type cells drawn on the paper's rings."""
    pos, typ = [], []
    for j in range(3):
        for i in range(1, int(n_type[j]) + 1):
            th = 2 * np.pi * i / int(n_type[j])
            pos.append(R_RAD[j] * np.array([np.sin(th), np.cos(th)]))
            typ.append(j)
    pos, typ = np.array(pos).T, np.array(typ)
    T = np.zeros((3, typ.size))
    T[typ, np.arange(typ.size)] = 1.0
    sen = foxhound(pos, T)
    return np.stack([sen[:, typ == i].mean(axis=1) for i in range(3)], axis=1)


def fate_short(types, target=N_TYPE):
    """The kind furthest below the target / 16, by fraction of the count."""
    cnt = np.array([(types == k).sum() for k in range(3)], float)
    return int(np.argmax(target / 16.0 - cnt / max(cnt.sum(), 1.0)))


def gap_dir(P, kin, cpos, rng):
    """Unit vector from cpos into the widest angular gap among the columns kin of P."""
    if kin.size >= 2:
        ang = np.sort(np.arctan2(P[1, kin] - cpos[1], P[0, kin] - cpos[0]))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        g = int(np.argmax(gaps))
        mid = ang[g] + gaps[g] / 2.0
        u = np.array([np.cos(mid), np.sin(mid)])
    elif kin.size == 1:
        u = cpos - P[:, kin[0]]
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else rng.standard_normal(2)
    else:
        u = rng.standard_normal(2)
    return u / np.linalg.norm(u)


def angles(Pg, n):
    """Each tissue's angle about the organ's core (tissue 0)."""
    return np.arctan2(Pg[1, :n] - Pg[1, 0], Pg[0, :n] - Pg[0, 0])


def rings(Pg):
    """0 for a tissue on the inner ring, 1 for the outer; the core counts as outer."""
    g = (np.hypot(Pg[0] - Pg[0, 0], Pg[1] - Pg[1, 0]) >= RING_SPLIT).astype(int)
    g[0] = 1
    return g


def ring_wave(phi, Pg, grp, g, m=None, skip=None):
    """Ring g's wave, phase ~ offset + m x angle: (tissues, winding, offset, order, angles).  The winding is
    read only from 3 or more tissues with no gap of 150 degrees; given m, the offset is read whatever the gap."""
    mem = (grp[:phi.size] == g) & (np.arange(phi.size) != 0)
    if skip is not None:
        mem &= ~skip
    th = angles(Pg, phi.size)
    n = int(mem.sum())
    if n == 0:
        return 0, None, None, 0.0, th
    gap = 2 * np.pi
    if n >= 2:
        ts = np.sort(th[mem])
        gap = float(np.max(np.diff(np.r_[ts, ts[0] + 2 * np.pi])))
    if m is None:
        o = np.argsort(th[mem])
        q = phi[mem][o]
        m = int(np.round(cw(np.roll(q, -1) - q).sum() / (2 * np.pi))) if (n >= 3 and gap < np.radians(150)) else None
        if m is None:
            return n, None, None, 0.0, th
    z = np.mean(np.exp(1j * (phi[mem] - m * th[mem])))
    return n, m, float(np.angle(z)), float(np.abs(z)), th


# ---- the layout ---------------------------------------------------------------------------
class Layout:
    """N cells, NG ensembles (every one a blanket); own[i] = the ensemble cell i belongs to.
    Per agent: 2 position + 3 secretion actions, 3 causes, 3 secreted + 3 sensed observations."""
    def __init__(self, N, NG_, own):
        self.N, self.NG = N, NG_
        self.own = np.asarray(own, dtype=int)
        self.sizes = dict(l=N, g=NG_, c=N, s=N, b=1)
        self.a_off, self.v_off, self.y_off = {}, {}, {}
        oa = ov = oy = 0
        for k in KEYS:
            n = self.sizes[k]
            self.a_off[k], self.v_off[k], self.y_off[k] = oa, ov, oy
            oa, ov, oy = oa + 5 * n, ov + 3 * n, oy + 6 * n
        self.na, self.nv, self.ny = oa, ov, oy

    def a_unvec(self, a):
        out = {}
        for k in KEYS:
            n, o = self.sizes[k], self.a_off[k]
            out[k] = (a[o:o + 2 * n].reshape(2, n, order="F"), a[o + 2 * n:o + 5 * n].reshape(3, n, order="F"))
        return out

    def a_vec(self, pairs):
        return np.concatenate([x.flatten(order="F") for k in KEYS for x in pairs[k]])

    def v_unvec(self, v):
        return {k: v[self.v_off[k]:self.v_off[k] + 3 * self.sizes[k]].reshape(3, self.sizes[k], order="F") for k in KEYS}

    def v_vec(self, d):
        return np.concatenate([d[k].flatten(order="F") for k in KEYS])

    def y_vec(self, blocks):
        return np.concatenate([x.flatten(order="F") for k in KEYS for x in blocks[k]])

    def hist_unvec(self, A_hist):
        """The action history split by block: {k: (positions (2, n, T), secretions (3, n, T))}."""
        return {k: (A_hist[self.a_off[k]:self.a_off[k] + 2 * self.sizes[k]].reshape(2, self.sizes[k], -1, order="F"),
                    A_hist[self.a_off[k] + 2 * self.sizes[k]:self.a_off[k] + 5 * self.sizes[k]].reshape(3, self.sizes[k], -1, order="F"))
                for k in KEYS}

    def hist_vec(self, Hb, T):
        return np.concatenate([Hb[k][j].reshape((2 if j == 0 else 3) * self.sizes[k], T, order="F") for k in KEYS for j in (0, 1)])


# ---- process Gg, model Mg -----------------------------------------------------------------
def Gg(a, L, gains=None):
    """gains: one per ensemble, scaling what its cells put into the organ's field (not its head count)."""
    A = L.a_unvec(a)
    N, NG_, own = L.N, L.NG, L.own
    l_pos, l_sec = A["l"]
    g_pos = A["g"][0]                       # its secretion is not read: heard through its cells
    c_pos, c_sec = A["c"]
    w = 1.0 / Q
    src = c_sec / np.maximum(c_sec.sum(axis=0, keepdims=True), EMIT_FLOOR)
    cg = g_pos[:, own] + CELL_SCALE * c_pos
    sw = src * w
    if gains is not None:
        sw = sw * gains[own][None, :]
    l_sen = np.zeros_like(l_sec)
    for k in range(NG_):
        idx = np.flatnonzero(own == k)
        l_sen[:, idx] = foxhound(l_pos[:, idx], l_sec[:, idx])
    c_sen = foxhound(cg, sw)
    g_sec = np.zeros((3, NG_))
    g_sen = np.zeros((3, NG_))
    for k in range(NG_):
        idx = np.flatnonzero(own == k)
        c_sen[:, idx] = c_sen[:, idx] - foxhound(cg[:, idx], sw[:, idx])      # the other blankets only
        g_sec[:, k] = src[:, idx].sum(axis=1) * w
        g_sen[:, k] = c_sen[:, idx].mean(axis=1)
    bsec = (g_sec / np.maximum(g_sec.sum(axis=0, keepdims=True), 1e-9)).sum(axis=1, keepdims=True) / Q
    return L.y_vec({"l": (l_sec, l_sen), "g": (g_sec, g_sen), "c": (c_sec, c_sen),
                    "s": (c_sec, np.zeros((3, N))), "b": (bsec, np.zeros((3, 1)))})


def Mg(v, L, P_sen, P_org):
    """P_sen: (3, 3, N), each cell its own blanket's fingerprint."""
    V = L.v_unvec(v)
    pl, pg, pc = softmax(V["l"]), softmax(V["g"]), softmax(V["c"])
    pG = pg[:, L.own]
    ex = np.exp(V["s"])
    vol = ex.sum(axis=0, keepdims=True)
    p = softmax(V["s"])
    return L.y_vec({"l": (pl, np.einsum("ijn,jn->in", P_sen, pl)), "g": (pg, P_org @ pg), "c": (pc, P_org @ pc),
                    "s": (ex, vol * (p - pG)),          # s_sen: consensus, observed as 0
                    "b": (softmax(V["b"]), np.zeros((3, 1)))})


# ---- the fixed matrices -------------------------------------------------------------------
def restriction(L):
    """Which observation's error each action answers: an agent's own block, plus
    up: a blanket's g rows answer to its cells' view actuators (the paper's credit);
    down: the s_sec row about cell i answers to cell i's view actuators (the self-model's)."""
    R = np.zeros((L.ny, L.na))
    for k in KEYS:
        n = L.sizes[k]
        I = np.eye(n)
        R1 = np.hstack([np.kron(I, np.ones((3, 2))), np.kron(I, np.ones((3, 3)))])
        R1 = np.vstack([R1, R1])
        R[L.y_off[k]:L.y_off[k] + 6 * n, L.a_off[k]:L.a_off[k] + 5 * n] = R1
    gy, oc, N, NG_ = L.y_off["g"], L.a_off["c"], L.N, L.NG
    for k in range(NG_):
        g_obs = np.r_[gy + 3 * k:gy + 3 * k + 3, gy + 3 * NG_ + 3 * k:gy + 3 * NG_ + 3 * k + 3]
        cells = np.flatnonzero(L.own == k)
        ck_act = np.concatenate([oc + 2 * cells, oc + 2 * cells + 1,
                                 oc + 2 * N + 3 * cells[:, None] + np.arange(3)[None, :]], axis=None)
        R[np.ix_(g_obs, ck_act)] = 1.0
    s_obs = L.y_off["s"]
    for i in range(N):
        ci = np.r_[oc + 2 * i:oc + 2 * i + 2, oc + 2 * N + 3 * i:oc + 2 * N + 3 * i + 3]
        R[np.ix_(np.arange(s_obs + 3 * i, s_obs + 3 * i + 3), ci)] = 1.0
    return R


class Fixed:
    def __init__(self, L, pprec_l, g_pprec):
        n, d = E_n + 1, E_d + 1
        self.L, self.n, self.d = L, n, d
        nv, ny, na = L.nv, L.ny, L.na
        nc = nv
        ne = ny + nc
        gy = ny
        gr = gy + 1
        self.nv, self.ny, self.na, self.nc, self.ne, self.gy, self.gr = nv, ny, na, nc, ne, gy, gr
        iV = spm_DEM_R(n, E_s)
        # prior precisions: floor, cells, fates, organ's core
        pc = np.full(nc, PRIOR_FLOOR)
        pc[L.v_off["l"]:L.v_off["l"] + 3 * L.N] = np.repeat(np.asarray(pprec_l, float), 3)
        pc[L.v_off["g"]:L.v_off["g"] + 3 * L.NG] = np.repeat(np.asarray(g_pprec, float), 3)
        pc[L.v_off["g"]:L.v_off["g"] + 3] = PRIOR_PREC
        mv = np.r_[np.full(ny, np.exp(4.0)), pc]
        gu = np.r_[np.full(gy, np.exp(-2.0)), np.zeros(ne - gy)]
        Rr = np.zeros((gr, na))
        Rr[:ny, :] = restriction(L)
        self.Pu = np.kron(iV[:d, :d], np.eye(nv) * np.exp(-8.0))
        self.Pa = np.eye(na) * np.exp(-8.0)
        self.dedy = spm_speye(ne, ny)
        self.dedc = -spm_speye(ne, nc, nc - ne)
        sk = lambda A, B: sp.kron(to_csr(A), to_csr(B), format="csr")
        self.iS_s = sk(iV, sp.diags(mv))
        self.iS_T = self.iS_s.T.tocsr()
        self.iG_T = sk(iV, sp.diags(gu)).T.tocsr()
        self.R_s = sk(spm_speye(n, 1, 0), Rr)
        Dv_m = sk(spm_speye(d, d, 1), sp.identity(nv))
        Dc = sk(spm_speye(d, d, 1), sp.identity(nc))
        Da = sp.csr_matrix((na, na))
        self.Du_s, self.Dc_s, self.Da_s = Dv_m, Dc, Da
        self.Dq_s = sp.block_diag([Dv_m, Dc, Da], format="csr")
        self.Dv_p_s = sk(spm_speye(n, n, 1), sp.identity(gr))
        self.Dp_s = sp.block_diag([self.Dv_p_s, self.Dv_p_s], format="csr")
        dydv = sk(np.eye(n), spm_speye(gy, gr))
        self.dE_dv_s = (sk(spm_speye(n, n), self.dedy) @ dydv).tocsr()
        self.dE_dc_s = sk(spm_speye(n, d), self.dedc)
        self.Pu_s, self.Pa_s = to_csr(self.Pu), to_csr(self.Pa)
        self.speye_n1 = to_csr(spm_speye(n, 1, 0))
        if n * ne <= 8000:
            self.logdet_iS = spm_logdet(np.kron(iV, np.diag(mv)))
        else:
            self.logdet_iS = ne * spm_logdet(iV) + n * float(np.sum(np.log(mv)))
        self.o_quv = 2 * n * gr
        self.o_quu = self.o_quv + d * nv
        self.o_qua = self.o_quu + d * nc
        self.n_tot = self.o_qua + na


# ---- the ADEM step ------------------------------------------------------------------------
def adem_step(fx, Gfun, Mfun, v, vd, a, A_hist, iY, prior_mean, dt=E_dt):
    """v, vd: causes and their motion; a: actions; A_hist: (na, T).  Returns the new v, vd, a,
    this step's observations y, and the free-energy terms."""
    n, d = fx.n, fx.d
    nv, nc, ne, gr, gy, na = fx.nv, fx.nc, fx.ne, fx.gr, fx.gy, fx.na
    qu_v = [v, vd]
    pu_a = [x.ravel() for x in spm_DEM_embed(A_hist, n, iY)]
    pu_z = [np.zeros(gr) for _ in range(n)]
    qu_u = [np.zeros(nc) for _ in range(n)]
    qu_u[0] = prior_mean
    dgda_1, g1 = spm_diff_vec(Gfun, pu_a[0])
    dg_da = np.zeros((gr, na))
    dg_da[:gy, :] = dgda_1
    pu_v = [np.concatenate([g1, np.zeros(1)]) + pu_z[0]]
    for i in range(1, n):
        pu_v.append(dg_da @ pu_a[i] + pu_z[i])
    qu_y = [pu_v[i][:gy] for i in range(n)]
    dgdv_1, g_pred = spm_diff_vec(Mfun, qu_v[0])
    dgdv = np.vstack([dgdv_1, -np.eye(nv)])
    E = np.zeros(n * ne)                                        # prediction errors
    E[:ne] = np.concatenate([qu_y[0], qu_v[0]]) - np.concatenate([g_pred, qu_u[0]])
    for i in range(1, n):
        vi = qu_v[i] if i < d else np.zeros(nv)
        E[i * ne:(i + 1) * ne] = fx.dedy @ qu_y[i] + fx.dedc @ qu_u[i] - dgdv @ vi
    dE_du = to_csr(-np.kron(spm_speye(n, d), dgdv))
    dgda_gen = sp.kron(fx.speye_n1, to_csr(dg_da), format="csr")
    dE_da = fx.dE_dv_s @ dgda_gen.multiply(fx.R_s)             # restriction
    A = (fx.iS_T @ dE_du).T.tocsr()
    B = (fx.iG_T @ dE_da).T.tocsr()
    AdEdu = (A @ dE_du).toarray()
    C = AdEdu + fx.Pu
    try:
        Cs = 0.5 * (C + C.T)
        tol = max(np.finfo(float).eps * np.linalg.norm(Cs, np.inf) * Cs.shape[0], np.exp(-32.0))
        Lc = cho_factor(Cs + np.eye(Cs.shape[0]) * tol, lower=True, check_finite=False)
        ld_quc = -2.0 * float(np.sum(np.log(np.diag(Lc[0]))))
    except (LinAlgError, ValueError):
        ld_quc = spm_logdet(spm_inv(C))
    u_states = np.concatenate(qu_v[:d])
    dVdu = -A @ E - fx.Pu_s @ u_states
    dVda = -B @ E - fx.Pa_s @ a
    dVdc = np.zeros(d * nc)
    dVduu = -AdEdu - fx.Pu
    dVdaa = -B @ dE_da - fx.Pa_s
    dVduv = -A @ fx.dE_dv_s
    dVduc = -A @ fx.dE_dc_s
    dVdua = -A @ dE_da
    dVdav = -B @ fx.dE_dv_s
    dVdau = -B @ dE_du
    dVdac = -B @ fx.dE_dc_s
    u = np.zeros(fx.n_tot)
    u[:n * gr] = np.concatenate(pu_v)
    u[n * gr:2 * n * gr] = np.concatenate(pu_z)
    u[fx.o_quv:fx.o_quv + d * nv] = np.concatenate(qu_v[:d])
    u[fx.o_quu:fx.o_quu + d * nc] = np.concatenate(qu_u[:d])
    u[fx.o_qua:] = a
    dFdu = np.zeros(fx.n_tot)
    dFdu[:2 * n * gr] = fx.Dp_s @ u[:2 * n * gr]
    dFdu[fx.o_quv:] = np.concatenate([dVdu, dVdc, dVda]) + fx.Dq_s @ u[fx.o_quv:]
    dFduu = sp.bmat([
        [None,  fx.Dv_p_s, None,                       None,     None],
        [None,  fx.Dv_p_s, None,                       None,     None],
        [dVduv, None,      fx.Du_s + to_csr(dVduu),    dVduc,    dVdua],
        [None,  None,      None,                       fx.Dc_s,  None],
        [dVdav, None,      dVdau,                      dVdac,    dVdaa + fx.Da_s]], format="csr")
    if not (np.all(np.isfinite(dFdu)) and np.all(np.isfinite(dFduu.data))):
        raise FloatingPointError("non-finite gradient or Jacobian")
    du = None
    for k in range(7):                                          # local-linearisation step, halved on failure
        try:
            cand = spm_dx(dFduu, dFdu, dt / 2 ** k)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.all(np.isfinite(cand)):
            du = cand
            break
    if du is None:
        raise FloatingPointError("integration step not finite")
    u = u + du
    v_new = u[fx.o_quv:fx.o_quv + nv]
    vd_new = u[fx.o_quv + nv:fx.o_quv + 2 * nv]
    a_new = u[fx.o_qua:]
    EiSE_t = float(E @ (fx.iS_s @ E))
    Jt = -0.5 * EiSE_t + ld_quc + 0.5 * fx.logdet_iS
    return v_new, vd_new, a_new, qu_y[0], Jt, EiSE_t, ld_quc


# ---- the run ------------------------------------------------------------------------------
def run(seed=0, nY=720, cuts=None, turns=True, pace=0.6, core_T=None):
    """Grow the organ for nY steps.
      cuts = (steps, n, what)   at each step, n whole blankets removed (never the organ's core), cells and
                                all: what = "random", or "inner" (the inner ring first, then others at random)
      turns                     turn-taking from AT["turns"]; pace: how far the core's clock moves its
                                threshold (0: not at all); core_T: the core clock's period (None: its own)"""
    rng = np.random.default_rng(seed)
    pos0, typ, P_A = build_priors()
    P_org = P_A.copy()
    P_org[1, 1] = ORGAN_FP_ACT
    P_B = form_fp(FORM_B)
    llr = np.log((FORM_B / 16.0) / (N_TYPE / 16.0))

    # seed: the paper's organ, its core kept; the zygote
    v0 = typ * ORGAN_BIAS + rng.standard_normal(typ.shape) / 8.0
    POSg = pos0 * POS_BIAS + rng.standard_normal(pos0.shape) * POS_NOISE
    Vg, POSg = v0[:, :1].copy(), POSg[:, :1].copy()
    Vl0 = np.zeros((3, 1))
    Vl0[0, :] = V_BIAS
    Vl = Vl0 + rng.standard_normal(Vl0.shape) / 8.0
    POSl = rng.standard_normal((2, 1)) * SPREAD
    Vc = np.tile(typ[:, [0]], (1, 1)) * V_BIAS + np.random.default_rng([seed, 7, 0]).standard_normal((3, 1)) / 8.0
    N = 1
    own = np.zeros(N, dtype=int)
    L = Layout(N, Vg.shape[1], own)
    Vs = picture(Vc)
    V = L.v_vec(dict(l=Vl, g=Vg, c=Vc, s=Vs, b=np.zeros((3, 1))))
    Vd = np.zeros_like(V)
    a = L.a_vec(dict(l=(POSl, softmax(Vl)), g=(POSg, softmax(Vg)), c=(POSl.copy(), softmax(Vc)),
                     s=(POSl.copy(), softmax(Vs)), b=(np.zeros((2, 1)), softmax(np.zeros((3, 1))))))
    A_hist = a[:, None].copy()

    prior_mean_l = Vl0.copy()
    pprec = np.full(N, PRIOR_PREC)
    g_pmean = np.zeros((3, L.NG))
    g_pprec = np.full(L.NG, PRIOR_FLOOR)
    g0_mean = np.array([V_BIAS, 0.0, 0.0])
    born = np.full(N, -10 ** 6)
    last_div = np.full(N, -10 ** 6)
    anchor = {0: 0}
    streak_p, streak_k = {0: 0}, {0: 0}
    last_parent_div, last_prune = {0: -10 ** 6}, {0: -10 ** 6}
    streak_o, last_organ_div = 0, -10 ** 6
    cut_at = set(cuts[0]) if cuts else set()
    dl, ob = None, None                                 # form belief (- A, + B); backup of the other ring's bit
    gains = None
    P_sen = None
    ck = dict(phi=None, omega=None, u=None, grp=None, born=None, voice=None, chi=None, chio=None, nref=None,
              cref=None, wind=np.zeros(2), copy=None, cw=None, rn=None, rlast=None, nu=None, thc=None,
              rng=np.random.default_rng(10007 + seed), walk=np.zeros(2), walk_x=np.zeros(2),
              walk_rng=np.random.default_rng(70011 + seed), on=0, U=np.zeros(2), turn_rng=np.random.default_rng(80011 + seed),
              n0=None)                                  # clocks, the core's copy, the walk, the turns

    def make_fixed():
        return Fixed(L, pprec, g_pprec)

    def prior_vec():
        z = np.zeros((3, N))
        pg_ = g_pmean.copy()
        pg_[:, 0] = g0_mean
        return L.v_vec(dict(l=prior_mean_l, g=pg_, c=z, s=z, b=np.zeros((3, 1))))

    def rebuild(Vb, Vdb, Ab, Hb, T):
        nonlocal L, V, Vd, a, A_hist, Gf, Mf, fx
        L = Layout(N, Vb["g"].shape[1], own)
        V, Vd, a = L.v_vec(Vb), L.v_vec(Vdb), L.a_vec(Ab)
        A_hist = L.hist_vec(Hb, T)
        Gf = lambda aa, L=L: Gg(aa, L, gains)
        Mf = lambda vv, L=L: Mg(vv, L, P_sen, P_org)
        fx = make_fixed()

    def add_cell(Vb, Vdb, Ab, Hb, T, newv, vd_c, i, dpos):
        """One cell's columns in every per-cell block: a daughter of cell i (offset dpos), or a zygote (i None)."""
        newv = dict(newv, s=picture(newv["c"]))
        for k in PER_CELL:
            sec = softmax(newv[k])
            Vb[k] = np.hstack([Vb[k], newv[k]])
            Vdb[k] = np.hstack([Vdb[k], np.zeros((3, 1)) if k == "l" else vd_c[k]])
            P_, S_ = Ab[k]
            Hp, Hs = Hb[k]
            if i is None:
                pos, hp = np.zeros((2, 1)), np.zeros((2, 1, T))
            else:
                pos, hp = P_[:, [i]] + dpos[:, None], Hp[:, [i], :] + dpos[None, :, None].transpose(1, 0, 2)
            Ab[k] = (np.hstack([P_, pos]), np.hstack([S_, sec]))
            Hb[k] = (np.concatenate([Hp, hp], axis=1), np.concatenate([Hs, np.repeat(sec[:, :, None], T, axis=2)], axis=1))

    def target(k):
        return (FORM_B.astype(float), PULL_R_B) if dl[k] >= 0.0 else (N_TYPE.astype(float), PULL_R)

    # a new clock's rate: remembered, else the copy's, else the mean
    def ring_rate(g, omega, grp, skip=None):
        if ck["nu"] is not None and not np.isnan(ck["nu"][g]):
            return float(ck["nu"][g])
        if ck["cw"] is not None and not np.isnan(ck["cw"][g]):
            return float(ck["cw"][g])
        mem = (grp[:omega.size] == g) & (np.arange(omega.size) != 0)
        if skip is not None:
            mem &= ~skip[:omega.size]
        return float(omega[mem].mean()) if mem.any() else None

    def onto_wave(phi, Pg, grp, g, k, skip=None):
        m = int(ck["wind"][g])
        n, _, p0, _, th = ring_wave(phi, Pg, grp, g, m=m, skip=skip)
        if n == 0:
            if ck["copy"] is None or np.isnan(ck["copy"][g]):
                return None
            p0 = float(ck["copy"][g])
        return p0 + m * th[k] if k < th.size else p0 + m * angles(Pg, k + 1)[k]

    Gf = lambda aa: Gg(aa, L, gains)
    Mf = lambda vv: Mg(vv, L, P_sen, P_org)
    fx = make_fixed()
    rec = dict(N=[], types=[], pos=[], sen=[], p=[], own=[], organ_types=[], organ_N=[], organ_pos=[], organ_emit=[],
               views=[], organ_sec=[], organ_sen=[], J=[], divisions=[], celled_births=[], organ_divisions=[], removals=[],
               organ_removals=[], form_dl=[], form_ob=[], clock_group=[], clock_omega=[], clock_phase=[], clock_drive=[],
               clock_hold=[], clock_shadow=[], clock_nu=[], access=[], access_x=[])
    EiSE = Hquc = 0.0
    t0 = time.time()
    for iY in range(1, nY + 1):
        vote = iY >= AT["vote"]
        # 0. forms: a new blanket takes its nearest's belief
        Pg = L.a_unvec(a)["g"][0]
        if dl is None:
            dl = np.full(L.NG, -8.0)
        elif dl.size < L.NG:
            old = dl.size
            if ob is not None and ob.size == old:
                rg = rings(Pg)
                if ck["grp"] is not None and ck["grp"].size >= old:
                    rg[:old] = ck["grp"][:old]
            new, new_ob = [], []
            for k_ in range(old, L.NG):
                dd = np.linalg.norm(Pg[:, :old] - Pg[:, [k_]], axis=0)
                if vote and old > 1:
                    dd[0] = np.inf
                src_ = int(np.argmin(dd))
                if ob is None or ob.size != old:
                    new.append(dl[src_])
                elif rg[src_] == rg[k_]:
                    new.append(dl[src_]); new_ob.append(ob[src_])
                else:
                    new.append(ob[src_]); new_ob.append(dl[src_])
            if ob is not None and ob.size == old:
                ob = np.r_[ob, new_ob]
            dl = np.r_[dl, new]
        bk = 1.0 / (1.0 + np.exp(-dl))
        P_sen = (1.0 - bk[own])[None, None, :] * P_A[:, :, None] + bk[own][None, None, :] * P_B[:, :, None]
        # 0b. clocks: a new clock starts on its ring's wave
        if iY >= AT["clocks"]:
            NGc = L.NG
            if ck["phi"] is None or ck["phi"].size != NGc:
                old = 0 if ck["phi"] is None else min(ck["phi"].size, NGc)
                u = ck["rng"].uniform(-1.0, 1.0, NGc - old)
                ph = ck["rng"].uniform(0.0, 2 * np.pi, NGc - old)
                g_new = rings(Pg)[old:NGc]
                om = 2 * np.pi / (np.where(g_new == 1, T_RING[1], T_RING[0]) * np.exp(T_SPREAD * u))
                if old > 0 and iY >= AT["walk"] and ck["grp"] is not None and ck["grp"].size >= old:
                    for q_, k_ in enumerate(range(old, NGc)):
                        p_ = onto_wave(ck["phi"][:old], Pg, ck["grp"][:old], int(g_new[q_]), k_)
                        if p_ is not None:
                            ph[q_] = p_
                    for q_ in range(NGc - old):
                        r_ = ring_rate(int(g_new[q_]), ck["omega"][:old], ck["grp"][:old])
                        if r_ is not None:
                            om[q_] = r_
                keep_ = lambda x, new_: new_ if (old == 0 or x is None) else np.r_[x[:old], new_]
                ck["grp"] = keep_(ck["grp"], g_new)
                rec["clock_group"].append((iY, ck["grp"].tolist()))
                ck["phi"], ck["u"] = keep_(ck["phi"], ph), keep_(ck["u"], u)
                ck["born"] = keep_(ck["born"], np.full(NGc - old, -10 ** 9 if old == 0 else iY))
                for k8 in ("nref", "cref"):
                    if ck[k8] is not None:
                        ck[k8] = np.r_[ck[k8][:old], np.zeros(NGc - old)]
                ck["voice"] = keep_(ck["voice"], np.full(NGc - old, old == 0))
                ck["omega"] = keep_(ck["omega"], om)
                rec["clock_omega"].append((iY, np.round(ck["omega"], 6).tolist()))
            g_now = rings(Pg)
            if ck["grp"].size == NGc and (g_now != ck["grp"]).any():
                ch_ = g_now != ck["grp"]
                ck["grp"] = g_now
                if vote and ob is not None and ob.size == NGc and dl.size == NGc:
                    dl[ch_], ob[ch_] = ob[ch_].copy(), dl[ch_].copy()
                    if ck["chi"] is not None and ck["chio"] is not None and ck["chi"].size == NGc and ck["chio"].size == NGc:
                        ck["chi"][ch_], ck["chio"][ch_] = ck["chio"][ch_].copy(), ck["chi"][ch_].copy()
                ck["omega"][ch_] = 2 * np.pi / (np.where(g_now[ch_] == 1, T_RING[1], T_RING[0]) * np.exp(T_SPREAD * ck["u"][ch_]))
                if iY >= AT["walk"]:
                    for k_ in np.flatnonzero(ch_):
                        if k_ == 0:
                            continue
                        r_ = ring_rate(int(g_now[k_]), ck["omega"], g_now, skip=ch_)
                        if r_ is not None:
                            ck["omega"][k_] = r_
                    rec["clock_omega"].append((iY, np.round(ck["omega"], 6).tolist()))
                    for k_ in np.flatnonzero(ch_):
                        if k_ == 0:
                            continue
                        p_ = onto_wave(ck["phi"], Pg, g_now, int(g_now[k_]), k_, skip=ch_)
                        if p_ is not None:
                            ck["phi"][k_] = p_
                rec["clock_group"].append((iY, ck["grp"].tolist()))
            gains = 1.0 + AMP * np.sin(ck["phi"])
            # 0c. turn-taking
            if turns and iY >= AT["turns"] and ck["grp"].size == gains.size:
                n_ = gains.size
                if core_T is not None:
                    ck["omega"][0] = 2 * np.pi / float(core_T)
                th_ = angles(Pg, n_)
                dla_ = np.r_[dl, np.zeros(max(0, n_ - dl.size))][:n_]
                oba_ = np.zeros(n_) if ob is None else np.r_[ob, np.zeros(max(0, n_ - ob.size))][:n_]
                ga_ = ck["grp"][:n_]
                bel_ = (np.where(ga_ == 0, dla_, oba_), np.where(ga_ == 1, dla_, oba_))
                nc_ = np.arange(n_) != 0
                if ck["n0"] is None:
                    ck["n0"] = np.array([max(1, int(np.sum((ga_ == g_) & nc_))) for g_ in (0, 1)])
                # strength: order along the held winding x share of clocks
                ev_ = []
                for g_ in (0, 1):
                    m_ = int(ck["wind"][g_]) if ck["wind"][g_] != 0 else None
                    nw_, mg_, _, Rw_, _ = ring_wave(ck["phi"][:n_], Pg, ga_, g_, m=m_)
                    ev_.append(float(Rw_) * min(1.0, nw_ / float(ck["n0"][g_])) if mg_ is not None else 0.0)
                k_ = int(ck["on"]); j_ = 1 - k_
                ck["U"][j_] += ev_[j_] * max(0.0, 1.0 + TURN_NOISE * ck["turn_rng"].standard_normal())
                if ck["U"][j_] >= TURN_THRESH * (1.0 - float(pace) * np.cos(ck["phi"][0])):
                    ck["on"] = j_; ck["U"][:] = 0.0; k_ = j_
                x_ = -np.sign(bel_[k_]) * (np.cos(th_) if k_ == 0 else np.sin(th_))
                x_[0] = 0.0
                gains = gains * (1.0 + TURN_EPS * x_)
                rec["access"].append((iY, k_, np.round(ck["U"], 4).tolist(), np.round(ev_, 4).tolist(),
                                      round(float(ck["phi"][0] % (2 * np.pi)), 4)))
                rec["access_x"].append((iY, np.round(x_, 4).tolist()))
        # 1. the ADEM step
        V, Vd, a, y, Jt, e2, ld = adem_step(fx, Gf, Mf, V, Vd, a, A_hist, iY, prior_vec())
        EiSE += e2
        Hquc += ld
        # 2. cores held in place
        for j in sorted(j for j in anchor.values() if j is not None):
            for k in PER_CELL:
                o = L.a_off[k] + 2 * j
                a[o:o + 2] = A_hist[o:o + 2, -1]
        o = L.a_off["g"]
        a[o:o + 2] = A_hist[o:o + 2, -1]
        # 3. cells pulled to their kind's radius
        t_now = softmax(L.v_unvec(V)["l"]).argmax(axis=0)
        ol = L.a_off["l"]
        Pl_now = a[ol:ol + 2 * N].reshape(2, N, order="F")
        for k_ in range(L.NG):
            idx = np.flatnonzero(own == k_)
            ci = anchor[k_]
            # no core: nearest core-believer, else centroid
            if ci is None:
                cen = Pl_now[:, idx].mean(axis=1)
                cores_ = idx[t_now[idx] == 0]
                ci = int(cores_[np.argmin(np.linalg.norm(Pl_now[:, cores_] - cen[:, None], axis=0))]) if cores_.size else -1
            cpos = Pl_now[:, ci] if ci >= 0 else Pl_now[:, idx].mean(axis=1)
            pr_ = target(k_)[1]
            for j in idx:
                if j == ci or t_now[j] == 0:
                    continue
                v_ = Pl_now[:, j] - cpos
                d_ = float(np.linalg.norm(v_))
                if d_ < 1e-6:
                    continue
                delta = PULL * (cpos + float(pr_[t_now[j] - 1]) * v_ / d_ - Pl_now[:, j])
                for k in PER_CELL:
                    o = L.a_off[k] + 2 * j
                    a[o:o + 2] += delta
        # 4. ensembles pulled around the organ's core
        gt_now = softmax(L.v_unvec(V)["g"]).argmax(axis=0)
        og = L.a_off["g"]
        Pg_now = a[og:og + 2 * L.NG].reshape(2, L.NG, order="F")
        cpos = Pg_now[:, 0]
        for k_ in range(L.NG):
            if k_ == 0 or gt_now[k_] == 0:
                continue
            v_ = Pg_now[:, k_] - cpos
            d_ = float(np.linalg.norm(v_))
            if d_ < 1e-6:
                continue
            o = og + 2 * k_
            a[o:o + 2] += PULL * (cpos + float(PULL_R[gt_now[k_] - 1]) * v_ / d_ - Pg_now[:, k_])
        # 5. history
        A_hist = np.hstack([A_hist, a[:, None]])

        # 6. record
        Vb, Ab = L.v_unvec(V), L.a_unvec(a)
        p = softmax(Vb["l"])
        types = p.argmax(axis=0)
        sen = y[L.y_off["l"] + 3 * N:L.y_off["l"] + 6 * N].reshape(3, N, order="F")
        pred = P_sen[:, types, np.arange(types.size)]
        expected = pred.sum(axis=0)
        cdef = np.clip(pred - sen, 0.0, None).sum(axis=0)
        cdef = np.where(expected > 1e-9, cdef / np.maximum(expected, 1e-9), 0.0)
        g_sec_all = y[L.y_off["g"]:L.y_off["g"] + 3 * L.NG].reshape(3, L.NG, order="F")
        g_sen_all = y[L.y_off["g"] + 3 * L.NG:L.y_off["g"] + 6 * L.NG].reshape(3, L.NG, order="F")
        pdef_by = {k_: 1.0 - float(g_sec_all[:, k_].sum()) for k_ in range(L.NG)}
        organ_emit = float(y[L.y_off["b"]:L.y_off["b"] + 3].sum())
        # a coreless blanket adopts a new core
        for k_ in sorted(anchor):
            if anchor[k_] is None and ((types == 0) & (own == k_)).any():
                anchor[k_] = int(np.flatnonzero((types == 0) & (own == k_))[0])
        rec["N"].append(int(N))
        rec["types"].append(types.tolist())
        rec["pos"].append(np.round(Ab["l"][0], 4).tolist())
        rec["sen"].append(np.round(sen, 4).tolist())
        rec["p"].append(np.round(p, 4).tolist())
        rec["own"].append(own.tolist())
        rec["organ_types"].append(softmax(Vb["g"]).argmax(axis=0).tolist())
        rec["organ_N"].append(int(L.NG))
        rec["organ_pos"].append(np.round(Ab["g"][0], 3).tolist())
        rec["organ_emit"].append(round(organ_emit, 4))
        rec["views"].append(softmax(Vb["c"]).argmax(axis=0).tolist())
        rec["organ_sec"].append(np.round(g_sec_all, 4).tolist())
        rec["organ_sen"].append(np.round(g_sen_all, 4).tolist())
        rec["J"].append(float(Jt))

        # 6b. forms: evidence, ring reading, backup
        Pg = Ab["g"][0]
        ek = np.zeros(dl.size)
        for k_ in range(L.NG):
            if k_ < dl.size and int((own == k_).sum()) >= int(round(Q)):
                ek[k_] = float(np.array([(types[own == k_] == j_).sum() for j_ in range(3)]) @ llr)
        read_ = 0.0
        if ck["phi"] is not None and ck["phi"].size == dl.size:
            n_ = dl.size
            th_ = angles(Pg, n_)
            dth_ = cw(th_[None, :] - th_[:, None]); dth_[0, :] = 0.0; dth_[:, 0] = 0.0
            dk_ = np.hypot(Pg[0, :n_][:, None] - Pg[0, :n_][None, :], Pg[1, :n_][:, None] - Pg[1, :n_][None, :])
            sg_ = ck["grp"] if ck["grp"].size == n_ else np.zeros(n_, dtype=int)
            wn_ = np.exp(-dk_ ** 2 / (2 * SIGMA ** 2))
            np.fill_diagonal(wn_, 0.0); wn_[0, :] = 0.0; wn_[:, 0] = 0.0
            wn_ = wn_ * (sg_[:, None] == sg_[None, :])
            sd_ = np.sin(dth_)
            loc_ = (wn_ * np.sin(ck["phi"][None, :] - ck["phi"][:, None]) * sd_).sum(axis=1) \
                / np.maximum((wn_ * sd_ ** 2).sum(axis=1), 1e-9)
            if vote:
                loc_ = np.clip(loc_, -1.0, 1.0)
            oth_ = (sg_ == sg_[0]) & (np.arange(n_) != 0)
            loc_[0] = np.mean(loc_[oth_]) if oth_.any() else 0.0
            nc_ = np.arange(n_) != 0
            ring_mean = lambda x: [x[(sg_ == g) & nc_].mean() if ((sg_ == g) & nc_).any() else 0.0 for g in (0, 1)]
            # pooled by ring from AT["vote"]; none under READ_MIN
            if vote:
                gv_ = ring_mean(loc_)
                loc_ = np.array([gv_[g] for g in sg_])
                cnt_ = np.array([((sg_ == g) & nc_).sum() for g in sg_])
                loc_ = np.where(cnt_ >= READ_MIN, loc_, 0.0)
            if ck["chi"] is None or ck["chi"].size != n_:
                old_ = 0 if ck["chi"] is None else min(ck["chi"].size, n_)
                ck["chi"] = np.r_[(ck["chi"][:old_] if ck["chi"] is not None else np.zeros(0)), np.zeros(n_ - old_)]
            ck["chi"] = ck["chi"] + (loc_ - ck["chi"]) / READ_TAU
            read_ = -READ * np.clip(ck["chi"], -1.0, 1.0)                           # +1 (anticlockwise) says A
            if vote:
                gv_ = ring_mean(loc_)
                to_ = np.array([gv_[1 - int(g)] for g in sg_])
                co_ = np.zeros(n_) if ck["chio"] is None else np.r_[ck["chio"][:n_], np.zeros(max(0, n_ - ck["chio"].size))]
                ck["chio"] = co_ + (to_ - co_) / READ_TAU
                ob_ = np.zeros(n_) if ob is None else np.r_[ob[:n_], np.zeros(max(0, n_ - ob.size))]
                ob = np.clip((1.0 - FORM_DECAY) * ob_ - READ * np.clip(ck["chio"], -1.0, 1.0), -20.0, 20.0)
                rec["form_ob"].append((iY, np.round(ob, 4).tolist()))
        dl = np.clip((1.0 - FORM_DECAY) * dl + FORM_KAPPA * ek + read_, -20.0, 20.0)
        rec["form_dl"].append(np.round(dl, 4).tolist())

        # 6c. clocks: coupling, tilt, write, walk, hold, carry; windings, copy
        if ck["phi"] is not None and ck["phi"].size == L.NG:
            phi, grp, n_ = ck["phi"], ck["grp"], ck["phi"].size
            dphi = ck["omega"].copy()
            same_ = (grp[:, None] == grp[None, :]).astype(float)
            np.fill_diagonal(same_, 0.0)
            if vote:                                                                # the core is not on the rings
                same_[0, :] = 0.0; same_[:, 0] = 0.0

            # no stronger links after a loss
            def strongest(k8, now_):
                if not vote:
                    return now_
                r8 = ck[k8]
                ck[k8] = now_ if (r8 is None or r8.size != now_.size) else np.maximum(now_, 0.99 * r8)
                return ck[k8]
            dif_ = phi[None, :] - phi[:, None]
            dk_ = np.hypot(Pg[0][:, None] - Pg[0][None, :], Pg[1][:, None] - Pg[1][None, :])
            w_ = same_ * (np.exp(-dk_ ** 2 / (2 * SIGMA ** 2)) - REPEL)
            dphi = dphi + (w_ * np.sin(dif_)).sum(axis=1) / strongest("nref", np.maximum(np.abs(w_).sum(axis=1), 1e-12))
            m_ch = None
            if dl.size == n_:
                s_ = np.where(dl < 0.0, 1.0, -1.0)
                speak_ = (iY - ck["born"]) >= QUIET                                 # a new tissue listens first
                chi_ = ck["chi"] if (ck["chi"] is not None and ck["chi"].size == n_) else np.zeros(n_)
                ck["voice"] = ck["voice"] | (speak_ & (np.abs(chi_) >= QUIET_UNTIL) & (np.sign(chi_) == s_))
                speak_ = ck["voice"].copy()
                m_ch = FEEDBACK * s_ * speak_
                # vote by ring; a ring with no say takes the other's backups
                if vote:
                    mk_ = (grp[:, None] == grp[None, :]) & speak_[None, :]
                    V_ = np.sign((mk_ * s_[None, :]).sum(axis=1))
                    if ob is not None and ob.size == n_:
                        mo_ = (grp[:, None] != grp[None, :]) & speak_[None, :]
                        V_ = np.where(mk_.any(axis=1), V_, np.sign((mo_ * -np.sign(ob)[None, :]).sum(axis=1)))
                    m_ch = FEEDBACK * V_
            if AT["clocks"] <= iY < AT["write"]:
                m_ch = WRITE[grp]
            if m_ch is not None:
                th_ = angles(Pg, n_)
                dth_ = cw(th_[None, :] - th_[:, None]); dth_[0, :] = 0.0; dth_[:, 0] = 0.0
                wp_ = same_ * np.exp(-dk_ ** 2 / (2 * SIGMA ** 2))
                dphi = dphi + np.abs(m_ch) * (wp_ * np.sin(dif_ - np.sign(m_ch)[:, None] * dth_)).sum(axis=1) \
                    / strongest("cref", np.maximum(wp_.sum(axis=1), 1e-12))
            # zero-sum: coupling moves phase within a ring
            if vote:
                c8_ = dphi - ck["omega"]
                for g_ in (0, 1):
                    m8_ = (grp == g_) & (np.arange(n_) != 0)
                    if m8_.any():
                        dphi[m8_] = dphi[m8_] - c8_[m8_].mean()
            dv_ = np.zeros(2)
            if iY >= AT["walk"]:
                a_ = 1.0 / WALK_TAU
                ck["walk"] = (1.0 - a_) * ck["walk"] + WALK_SPEED * np.sqrt(2.0 * a_) * ck["walk_rng"].standard_normal(2)
                ck["walk_x"] = ck["walk_x"] + ck["walk"]
                dphi = dphi + 2 * np.pi * ck["walk"][grp]
                dv_ = ck["walk"].copy()
                rec["clock_drive"].append((iY, np.round(ck["walk"], 6).tolist(), np.round(ck["walk_x"], 6).tolist()))
                if ck["copy"] is not None and ck["rn"] is not None:                 # a regrowing ring is held to the copy
                    th8, on8 = angles(Pg, n_), [0, 0]
                    for g_ in (0, 1):
                        if np.isnan(ck["copy"][g_]) or (iY - ck["rlast"][g_]) >= STABLE:
                            continue
                        mem8 = (grp == g_) & (np.arange(n_) != 0)
                        tg8 = ck["copy"][g_] + ck["wind"][g_] * th8
                        dphi[mem8] = dphi[mem8] + HOLD_PULL * np.sin(tg8[mem8] - phi[mem8])
                        on8[g_] = 1
                    rec["clock_hold"].append((iY, on8))
            ck["phi"] = phi = phi + dphi
            # carry: moving round the core takes the wave's phase there
            if vote:
                thc_ = angles(Pg, n_)
                if ck["thc"] is not None and ck["thc"].size == n_:
                    mv_ = cw(thc_ - ck["thc"])
                    mv_ = np.where(np.abs(mv_) < 0.3, mv_, 0.0); mv_[0] = 0.0
                    ck["phi"] = phi = phi + ck["wind"][grp] * mv_
                ck["thc"] = thc_
            rec["clock_phase"].append((iY, np.round(phi, 5).tolist()))
            if ck["copy"] is None:
                ck["copy"], ck["cw"] = np.full(2, np.nan), np.full(2, np.nan)
            # windings; the copy is a phase-locked loop on the offset
            for g_ in (0, 1):
                n8, m_, p0_, R_, _ = ring_wave(phi, Pg, grp, g_)
                if m_ is not None:
                    ck["wind"][g_] = m_
                ok_ = m_ is not None and R_ >= 0.5
                if np.isnan(ck["copy"][g_]):
                    if ok_:
                        ck["copy"][g_] = p0_; ck["cw"][g_] = 2 * np.pi / T_RING[g_]
                    continue
                pr_ = ck["copy"][g_] + ck["cw"][g_] + 2 * np.pi * dv_[g_]
                if ck["rn"] is None:
                    ck["rn"], ck["rlast"] = np.full(2, -1), np.zeros(2)
                if n8 != ck["rn"][g_]:
                    ck["rn"][g_], ck["rlast"][g_] = n8, iY
                stable_ = (iY - ck["rlast"][g_]) >= STABLE
                e_ = float(cw(p0_ - pr_)) if ok_ else 0.0
                if iY >= AT["walk"] and not stable_:
                    e_ = 0.0
                ck["copy"][g_] = float(cw(pr_ + COPY_GAIN * e_))
                if stable_:
                    m9 = (grp == g_) & (np.arange(n_) != 0)
                    if m9.any():
                        if ck["nu"] is None:
                            ck["nu"] = np.full(2, np.nan)
                        ck["nu"][g_] = float(ck["omega"][m9].mean())
                    ck["cw"][g_] += COPY_GAIN ** 2 / 4.0 * e_
            if iY >= AT["walk"]:
                for g_ in (0, 1):
                    if np.isnan(ck["cw"][g_]):
                        continue
                    tg9 = ck["nu"][g_] if (ck["nu"] is not None and not np.isnan(ck["nu"][g_])) else ck["cw"][g_]
                    m9 = (grp == g_) & (np.arange(n_) != 0)
                    ck["omega"][m9] += (tg9 - ck["omega"][m9]) / ADAPT
                if iY % 10 == 0:
                    rec["clock_omega"].append((iY, np.round(ck["omega"], 6).tolist()))
            if ck["nu"] is not None and iY % 10 == 0:
                rec["clock_nu"].append((iY, np.round(ck["nu"], 7).tolist()))
            rec["clock_shadow"].append((iY, np.round(ck["copy"], 5).tolist(), np.round(ck["cw"], 6).tolist(),
                                        ck["wind"].astype(int).tolist()))

        # 7. blanket births
        new = []
        eligible = (iY - last_div >= REFRACTORY) & (iY - born >= REFRACTORY)
        for k_ in range(L.NG):
            streak_p[k_] = streak_p[k_] + 1 if pdef_by[k_] > THETA else 0
            elig_k = eligible & (own == k_)
            if streak_p[k_] >= TAU and iY - last_parent_div[k_] >= REFRACTORY and elig_k.any():
                cand = np.flatnonzero(elig_k)
                fate = fate_short(types[own == k_], target(k_)[0])
                inner = cand[types[cand] == max(fate - 1, 0)]
                pool = inner if inner.size else cand
                new.append(int(pool[np.argmax(cdef[pool] + 1e-9 * rng.random(pool.size))]))
                last_parent_div[k_] = iY
                streak_p[k_] = 0
        if new:
            Vb_ = {k: Vb[k].copy() for k in KEYS}
            Vdb = L.v_unvec(Vd)
            Vdb_ = {k: Vdb[k].copy() for k in KEYS}
            Ab_ = {k: (Ab[k][0].copy(), Ab[k][1].copy()) for k in KEYS}
            Hn = {k: (h[0].copy(), h[1].copy()) for k, h in L.hist_unvec(A_hist).items()}
            T = A_hist.shape[1]
            Pl = Ab_["l"][0]
            for i in new:
                sib = np.flatnonzero(own[:N] == own[i])
                fk = fate_short(types[sib], target(int(own[i]))[0])
                if sib.size > 1 and fk in (1, 2):
                    # on the BIRTH_R circle, in her kind's widest gap
                    cen = Pl[:, sib].mean(axis=1)
                    cores = sib[types[sib] == 0]
                    cpos = Pl[:, cores[np.argmin(np.linalg.norm(Pl[:, cores] - cen[:, None], axis=0))]] if cores.size else cen
                    u = gap_dir(Pl, sib[types[sib] == fk], cpos, rng)
                    dpos = cpos + BIRTH_R * u - Pl[:, i]
                else:
                    u = rng.standard_normal(2)
                    u = u / np.linalg.norm(u)
                    dpos = (OFFSET_SKIN if fk == 2 else OFFSET) * u
                vl_new = np.zeros((3, 1))
                vl_new[fk, 0] = V_BIAS
                vl_new = vl_new + rng.standard_normal((3, 1)) / 8.0
                vc_new = Vb_["c"][:, [i]] + rng.standard_normal((3, 1)) / 8.0
                add_cell(Vb_, Vdb_, Ab_, Hn, T, {"l": vl_new, "c": vc_new}, {k: Vdb_[k][:, [i]] for k in ("c", "s")}, i, dpos)
                born = np.append(born, iY)
                last_div = np.append(last_div, iY)
                last_div[i] = iY
                own = np.append(own, own[i])
                prior_mean_l = np.hstack([prior_mean_l, np.zeros((3, 1))])
                pprec = np.append(pprec, PRIOR_FLOOR)
                rec["divisions"].append((iY, int(i), int(Vb_["l"].shape[1] - 1), int(types[i])))
            N = Vb_["l"].shape[1]
            rebuild(Vb_, Vdb_, Ab_, Hn, T)

        # 7b. turnover
        gone = []
        for k_ in range(L.NG):
            cells_k = np.flatnonzero(own[:types.size] == k_)
            if cells_k.size <= 1:
                continue
            ty_k = types[cells_k]
            surplus = np.array([(ty_k == k).sum() for k in range(3)], float) - target(k_)[0]
            streak_k[k_] = streak_k[k_] + 1 if (pdef_by[k_] <= THETA and surplus.max() > SURPLUS) else 0
            if streak_k[k_] >= TAU and iY - last_prune[k_] >= REFRACTORY:
                pool = cells_k[ty_k == int(np.argmax(surplus))]
                pool = pool[pool != anchor[k_]]
                if pool.size:
                    j = int(pool[np.argmax(cdef[pool] + 1e-9 * rng.random(pool.size))])
                    gone.append(j)
                    rec["removals"].append((iY, j, int(types[j]), "turnover"))
                    last_prune[k_] = iY
                streak_k[k_] = 0
        if gone:
            keep = np.array([j for j in range(N) if j not in gone])
            Vb, Vdb, Ab = L.v_unvec(V), L.v_unvec(Vd), L.a_unvec(a)
            Hb = L.hist_unvec(A_hist)
            T = A_hist.shape[1]
            for k in PER_CELL:
                Vb[k], Vdb[k] = Vb[k][:, keep], Vdb[k][:, keep]
                Ab[k] = (Ab[k][0][:, keep], Ab[k][1][:, keep])
                Hb[k] = (Hb[k][0][:, keep, :], Hb[k][1][:, keep, :])
            born, last_div, pprec = born[keep], last_div[keep], pprec[keep]
            prior_mean_l = prior_mean_l[:, keep]
            for k_ in anchor:
                anchor[k_] = None if anchor[k_] is None or anchor[k_] not in keep else int(np.flatnonzero(keep == anchor[k_])[0])
            own = own[keep]
            N = keep.size
            rebuild(Vb, Vdb, Ab, Hb, T)
            for k_ in streak_p:
                streak_p[k_] = 0

        # 7c. cuts (never the organ's core)
        organ_changed = False
        if iY in cut_at:
            Vb, Vdb, Ab = L.v_unvec(V), L.v_unvec(Vd), L.a_unvec(a)
            Hb = L.hist_unvec(A_hist)
            T = A_hist.shape[1]
            pool = np.arange(1, L.NG)
            if cuts[2] == "inner":
                Pg = Ab["g"][0]
                rr_ = np.hypot(Pg[0, pool] - Pg[0, 0], Pg[1, pool] - Pg[1, 0])
                inn_, rest_ = pool[rr_ < RING_SPLIT], pool[rr_ >= RING_SPLIT]
                ggone = [int(j) for j in inn_] + [int(j) for j in rng.choice(rest_, min(max(0, cuts[1] - inn_.size), rest_.size), replace=False)]
            else:
                ggone = [int(j) for j in rng.choice(pool, min(cuts[1], pool.size), replace=False)]
            gtypes = softmax(Vb["g"]).argmax(axis=0)
            for j in ggone:
                rec["organ_removals"].append((iY, j, int(gtypes[j]), "ablate"))
            keep = np.array([j for j in range(L.NG) if j not in ggone])
            remap = {int(old): new for new, old in enumerate(keep.tolist())}
            dl = dl[keep[keep < dl.size]]
            if ob is not None:
                ob = ob[keep[keep < ob.size]]
            if ck["phi"] is not None:
                kc = keep[keep < ck["phi"].size]
                for k8 in ("phi", "omega", "grp", "u", "chi", "chio", "born", "nref", "cref", "voice"):
                    ck[k8] = None if ck[k8] is None else ck[k8][kc]
            Vb["g"], Vdb["g"] = Vb["g"][:, keep], Vdb["g"][:, keep]
            Ab["g"] = (Ab["g"][0][:, keep], Ab["g"][1][:, keep])
            Hb["g"] = (Hb["g"][0][:, keep, :], Hb["g"][1][:, keep, :])
            g_pmean, g_pprec = g_pmean[:, keep], g_pprec[keep]
            ckeep = np.flatnonzero(np.isin(own, keep))
            for k in PER_CELL:
                Vb[k], Vdb[k] = Vb[k][:, ckeep], Vdb[k][:, ckeep]
                Ab[k] = (Ab[k][0][:, ckeep], Ab[k][1][:, ckeep])
                Hb[k] = (Hb[k][0][:, ckeep, :], Hb[k][1][:, ckeep, :])
            born, last_div, pprec = born[ckeep], last_div[ckeep], pprec[ckeep]
            prior_mean_l = prior_mean_l[:, ckeep]
            for d_ in (anchor, streak_p, last_parent_div, streak_k, last_prune):
                for kk in list(d_):
                    if kk not in remap:
                        del d_[kk]
            for kk in anchor:
                anchor[kk] = int(np.flatnonzero(ckeep == anchor[kk])[0]) if anchor[kk] is not None else None
            for d_ in (anchor, streak_p, last_parent_div, streak_k, last_prune):
                for kk in sorted(d_):
                    if remap[kk] != kk:
                        d_[remap[kk]] = d_.pop(kk)
            own = np.array([remap[int(o)] for o in own[ckeep]], dtype=int)
            N = ckeep.size
            rebuild(Vb, Vdb, Ab, Hb, T)
            streak_o, organ_changed = 0, True

        # 8. organ births
        streak_o = 0 if organ_changed else (streak_o + 1 if 1.0 - organ_emit > THETA else 0)
        if streak_o >= TAU and iY - last_organ_div >= REFRACTORY:
            Vb, Vdb, Ab = L.v_unvec(V), L.v_unvec(Vd), L.a_unvec(a)
            Hb = L.hist_unvec(A_hist)
            T = A_hist.shape[1]
            Pg, Sg = Ab["g"]
            gtypes = softmax(Vb["g"]).argmax(axis=0)
            fate_g = fate_short(gtypes)
            inner = np.flatnonzero(gtypes == max(fate_g - 1, 0))
            pool = inner if inner.size else np.arange(L.NG)
            im = int(rng.choice(pool)) if pool.size > 1 else int(pool[0])
            if fate_g in (1, 2):
                cen = Pg.mean(axis=1)
                gcores = np.flatnonzero(gtypes == 0)
                cpos = Pg[:, 0] if gtypes[0] == 0 else (Pg[:, gcores[np.argmin(np.linalg.norm(Pg[:, gcores] - cen[:, None], axis=0))]] if gcores.size else cen)
                u = gap_dir(Pg, np.flatnonzero(gtypes == fate_g), cpos, rng)
                dpos = cpos + BIRTH_R * u - Pg[:, im]
            else:                                                            # a lost core regrows in the middle
                dpos = Pg.mean(axis=1) - Pg[:, im]
            vg_new = np.zeros((3, 1))
            vg_new[fate_g, 0] = ORGAN_BIAS
            vg_new = vg_new + rng.standard_normal((3, 1)) / 8.0
            Hp, Hs = Hb["g"]
            Hb["g"] = (np.concatenate([Hp, Hp[:, [im], :] + dpos[None, :, None].transpose(1, 0, 2)], axis=1),
                       np.concatenate([Hs, np.repeat(softmax(vg_new)[:, :, None], T, axis=2)], axis=1))
            Vb["g"] = np.hstack([Vb["g"], vg_new])
            Vdb["g"] = np.hstack([Vdb["g"], np.zeros((3, 1))])
            Ab["g"] = (np.hstack([Pg, Pg[:, [im]] + dpos[:, None]]), np.hstack([Sg, softmax(vg_new)]))
            gfm = np.zeros((3, 1))
            gfm[fate_g, 0] = V_BIAS
            g_pmean = np.hstack([g_pmean, gfm])
            g_pprec = np.append(g_pprec, ORGAN_FATE_PREC)
            knew = int(L.NG)
            rec["organ_divisions"].append((iY, im, knew, int(gtypes[im]), fate_g))
            last_organ_div = iY
            streak_o = 0
            vl0 = np.zeros((3, 1))
            vl0[0, 0] = V_BIAS
            vc0 = np.zeros((3, 1))
            vc0[fate_g, 0] = ORGAN_BIAS
            newv = {"l": vl0 + rng.standard_normal((3, 1)) / 8.0, "c": vc0 + rng.standard_normal((3, 1)) / 8.0}
            add_cell(Vb, Vdb, Ab, Hb, T, newv, {"c": np.zeros((3, 1)), "s": np.zeros((3, 1))}, None, None)
            born = np.append(born, iY)
            last_div = np.append(last_div, iY)
            prior_mean_l = np.hstack([prior_mean_l, vl0])
            pprec = np.append(pprec, PRIOR_PREC)
            own = np.append(own, knew)
            N = N + 1
            anchor[knew], streak_p[knew], last_parent_div[knew] = N - 1, 0, -10 ** 6
            streak_k[knew], last_prune[knew] = 0, -10 ** 6
            rec["celled_births"].append((iY, knew, int(fate_g)))
            rebuild(Vb, Vdb, Ab, Hb, T)

    n, ny = fx.n, fx.ny
    F = -EiSE / 2.0 + Hquc / 2.0 + fx.logdet_iS * nY / 2.0 - n * ny * nY * np.log(2 * np.pi) / 2.0
    Vb, Ab = L.v_unvec(V), L.a_unvec(a)
    return dict(rec=rec, V=Vb, A=Ab, F=float(F), own=own.tolist(), NG=int(L.NG), N_final=int(N),
                seconds=round(time.time() - t0, 1),
                params=dict(seed=seed, nY=nY, cuts=cuts, turns=turns, pace=pace, core_T=core_T))
