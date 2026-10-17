"""Extended-study simulation engine (see PROTOCOL_extended_study.md).

Speed: each (feature map, n, noise) circuit template U(x)^dag-free form
U(z)^dag U(x) is transpiled once with free parameters and then evaluated
for every data pair through Aer parameter binds. Ideal execution uses the
exact statevector kernel plus binomial shot sampling.

Per configuration (map, n, dataset, seed, noise, c) we store the training
off-diagonal block, the measured diagonal, swapped estimates for a fixed
pair subset, and the test-train block, then score every selection rule.

Usage: python ext_engine.py <arm>   (arms defined in ARMS below)
"""
import json
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit.circuit.library import RXGate, RZZGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, coherent_unitary_error,
                              depolarizing_error)
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine, make_circles, make_moons)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "extended")
BANDWIDTHS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
SHOTS = 1024
N_SWAP = 200


# ------------------------------------------------------------ feature maps
def hea_map(n, reps=2):
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for q in range(n):
            qc.ry(x[q], q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


def feature_map(name, n):
    if name == "zz":
        return ZZFeatureMap(n, reps=2, entanglement="linear")
    if name == "yzz":
        return PauliFeatureMap(n, reps=2, paulis=["Y", "ZZ"],
                               entanglement="linear")
    if name == "hea":
        return hea_map(n)
    raise ValueError(name)


# ----------------------------------------------------------------- datasets
def quantum_native(n, n_samples, seed, c_true=0.25, margin=0.15):
    """Labels from a random parity observable on ZZ-feature-map states at a
    known bandwidth c_true (in the scaled [0, pi] coordinates)."""
    from qiskit.quantum_info import random_unitary, SparsePauliOp
    rng = np.random.default_rng(1000 + seed)
    fm = feature_map("zz", n)
    if n <= 12:
        V = random_unitary(2 ** n, seed=2000 + seed)
        obs = SparsePauliOp("Z" * n).to_matrix()
        M = V.adjoint().data @ obs @ V.data
        value = lambda psi: float(np.real(np.vdot(psi, M @ psi)))
    else:
        # a dense 2^n unitary does not fit in memory; use a random
        # depth-4 RY+CZ circuit as V and measure the Z-parity
        vr = np.random.default_rng(2000 + seed)
        Vc = QuantumCircuit(n)
        for _ in range(4):
            for q in range(n):
                Vc.ry(vr.uniform(0, 2 * np.pi), q)
            for q in range(n - 1):
                Vc.cz(q, q + 1)
        parity = np.array([(-1) ** bin(k).count("1") for k in range(2 ** n)])

        def value(psi):
            out = Statevector(psi).evolve(Vc).data
            return float(np.sum(parity * np.abs(out) ** 2))
    if n > 12:
        # parity values concentrate near zero at this width: take the most
        # separated points of each sign from a candidate pool instead of a
        # fixed margin
        pool = rng.uniform(0, np.pi, (8 * n_samples, n))
        vals = np.array([value(Statevector(fm.assign_parameters(c_true * x))
                               .data) for x in pool])
        half = n_samples // 2
        pos = np.argsort(-vals)[:half]
        neg = np.argsort(vals)[:half]
        idx = np.concatenate([pos, neg])
        return pool[idx], (vals[idx] > 0).astype(int)
    X, y, per = [], [], {0: 0, 1: 0}
    half = n_samples // 2
    while len(X) < 2 * half:
        x = rng.uniform(0, np.pi, n)
        psi = Statevector(fm.assign_parameters(c_true * x)).data
        val = value(psi)
        lab = int(val > 0)
        if abs(val) >= margin and per[lab] < half:
            X.append(x)
            y.append(lab)
            per[lab] += 1
    return np.array(X), np.array(y)


def load(name, n, n_train, n_test, seed):
    """Stratified split first; PCA and [0, pi] scaling fitted on training
    data only."""
    rng = np.random.RandomState(seed)
    total = n_train + n_test
    if name == "qnative":
        X, y = quantum_native(n, total, seed)
    else:
        if name == "moons":
            X, y = make_moons(n_samples=total, noise=0.2, random_state=seed)
        elif name == "circles":
            X, y = make_circles(n_samples=total, noise=0.1, factor=0.5,
                                random_state=seed)
        elif name == "breast_cancer":
            d = load_breast_cancer(); X, y = d.data, d.target
        elif name == "wine":
            d = load_wine(); m = d.target < 2; X, y = d.data[m], d.target[m]
        elif name == "iris":
            d = load_iris(); m = d.target > 0; X, y = d.data[m], d.target[m] - 1
        elif name.startswith("digits"):
            a, b = int(name[-3]), int(name[-1])  # e.g. digits3v5
            d = load_digits(); m = (d.target == a) | (d.target == b)
            X, y = d.data[m], (d.target[m] == b).astype(int)
        else:
            raise ValueError(name)
        if len(X) > total:
            idx = rng.choice(len(X), total, replace=False)
            X, y = X[idx], y[idx]
    n_te = min(n_test, len(X) - n_train) if len(X) < total else n_test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=n_te,
                                          random_state=seed, stratify=y)
    Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    if name != "qnative":
        if Xtr.shape[1] > n:
            p = PCA(n_components=n, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        elif Xtr.shape[1] < n:
            pad = n - Xtr.shape[1]
            Xtr = np.hstack([Xtr, np.zeros((len(Xtr), pad))])
            Xte = np.hstack([Xte, np.zeros((len(Xte), pad))])
        sc = MinMaxScaler((0, np.pi)).fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), np.clip(sc.transform(Xte), 0, np.pi)
    return Xtr, Xte, ytr, yte


# -------------------------------------------------------------------- noise
def simulator(noise, n, seed):
    if noise == "dep":
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1),
                                       ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
        sim, basis = AerSimulator(noise_model=nm), ["rz", "sx", "x", "cx"]
    elif noise in ("coh", "cohweak"):
        rx, rzz = (0.04, 0.15) if noise == "coh" else (0.015, 0.05)
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(
            depolarizing_error(0.002, 1).compose(
                coherent_unitary_error(RXGate(rx).to_matrix())), ["sx"])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1),
                                       ["rz", "x"])
        nm.add_all_qubit_quantum_error(
            depolarizing_error(0.02, 2).compose(
                coherent_unitary_error(RZZGate(rzz).to_matrix())), ["cx"])
        sim, basis = AerSimulator(noise_model=nm), ["rz", "sx", "x", "cx"]
    elif noise == "device":
        from qiskit_ibm_runtime.fake_provider import (FakeManilaV2,
                                                      FakeGuadalupeV2)
        b = FakeManilaV2() if n <= 5 else FakeGuadalupeV2()
        sim, basis = AerSimulator.from_backend(b), None
    else:
        raise ValueError(noise)
    sim.set_options(seed_simulator=seed, max_parallel_threads=0)
    return sim, basis


def template(fm, n, sim, basis, seed):
    px = ParameterVector("px", fm.num_parameters)
    pz = ParameterVector("pz", fm.num_parameters)
    qc = QuantumCircuit(n)
    qc.compose(fm.assign_parameters(px), inplace=True)
    qc.compose(fm.assign_parameters(pz).inverse(), inplace=True)
    qc.measure_all()
    t = transpile(qc, sim, basis_gates=basis, optimization_level=1,
                  seed_transpiler=seed)
    return t, px, pz


def run_pairs(t, px, pz, sim, A, B, n, shots, chunk=2000):
    """All-zeros frequency of U(B_k)^dag U(A_k) for every row k."""
    out = []
    zero = "0" * n
    for s in range(0, len(A), chunk):
        a, b = A[s:s + chunk], B[s:s + chunk]
        binds = {**{p: list(a[:, i]) for i, p in enumerate(px)},
                 **{p: list(b[:, i]) for i, p in enumerate(pz)}}
        res = sim.run(t, parameter_binds=[binds], shots=shots).result()
        for k in range(len(a)):
            out.append(res.get_counts(k).get(zero, 0) / shots)
    return np.array(out)


# -------------------------------------------------------------- one config
def estimate(fm, n, noise, Xtr, Xte, c, seed, swap_idx, cache):
    """Return dict of Gram blocks for one bandwidth."""
    rng = np.random.default_rng(seed)
    N, M = len(Xtr), len(Xte)
    iu = np.triu_indices(N, 1)
    A, B = c * Xtr, c * Xte
    if noise == "ideal":
        sv = lambda X: np.array([Statevector(fm.assign_parameters(x)).data
                                 for x in X])
        Str, Ste = sv(A), sv(B)
        Ktr_ex = np.abs(Str.conj() @ Str.T) ** 2
        Kte_ex = np.abs(Ste.conj() @ Str.T) ** 2
        off = rng.binomial(SHOTS, np.clip(Ktr_ex[iu], 0, 1)) / SHOTS
        sw = rng.binomial(SHOTS, np.clip(Ktr_ex[iu][swap_idx], 0, 1)) / SHOTS
        diag = np.ones(N)
        te = rng.binomial(SHOTS, np.clip(Kte_ex, 0, 1)) / SHOTS
    else:
        key = (noise, n)
        if key not in cache:
            sim, basis = simulator(noise, n, seed)
            cache[key] = (sim,) + template(fm, n, sim, basis, seed)
        sim, t, px, pz = cache[key]
        i, j = iu
        # forward U(x_j)^dag U(x_i), swapped U(x_i)^dag U(x_j), diagonal,
        # test rows U(x_tr)^dag U(x_te)
        Aall = np.vstack([A[i], A[j[swap_idx]], A,
                          np.repeat(B, N, axis=0)])
        Ball = np.vstack([A[j], A[i[swap_idx]], A, np.tile(A, (M, 1))])
        p = run_pairs(t, px, pz, sim, Aall, Ball, n, SHOTS)
        n1, n2 = len(i), len(swap_idx)
        off, sw = p[:n1], p[n1:n1 + n2]
        diag, te = p[n1 + n2:n1 + n2 + N], p[n1 + n2 + N:].reshape(M, N)
    return {"off": off, "swap": sw, "diag": diag, "te": te}


def run_config(arm, fmap, n, dataset, seed, noise, n_train, n_test):
    path = os.path.join(OUT, arm, f"{fmap}_n{n}_{dataset}_s{seed}_{noise}.npz")
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    Xtr, Xte, ytr, yte = load(dataset, n, n_train, n_test, seed)
    fm = feature_map(fmap, n)
    N = len(Xtr)
    rng = np.random.default_rng(seed)
    n_pairs = N * (N - 1) // 2
    swap_idx = np.sort(rng.choice(n_pairs, min(N_SWAP, n_pairs),
                                  replace=False))
    cache, blocks = {}, {}
    for c in BANDWIDTHS:
        r = estimate(fm, n, noise, Xtr, Xte, c, seed, swap_idx, cache)
        for k, v in r.items():
            blocks[f"{k}_c{c}"] = v
    np.savez_compressed(path, ytr=ytr, yte=yte, swap_idx=swap_idx,
                        bandwidths=np.array(BANDWIDTHS), **blocks)
    print(f"[{arm}] {fmap} n{n} {dataset} s{seed} {noise}: "
          f"{time.time() - t0:.0f}s", flush=True)


ARMS = {
    # 4 qubits: 8 datasets x 3 maps x 4 noise settings x 8 seeds
    "q4": dict(n=4, n_train=80, n_test=40, seeds=range(8),
               fmaps=["zz", "yzz", "hea"],
               datasets=["moons", "circles", "breast_cancer", "wine", "iris",
                         "digits3v5", "digits1v7", "qnative"],
               noises=["ideal", "dep", "cohweak", "coh", "device"]),
    # 8 qubits: ZZ map, 4 datasets x 4 noise settings x 6 seeds
    "q8": dict(n=8, n_train=60, n_test=30, seeds=range(6), fmaps=["zz"],
               datasets=["breast_cancer", "digits3v5", "wine", "qnative"],
               noises=["ideal", "dep", "coh", "device"]),
    # 12 qubits: ZZ map, 2 datasets x 3 noise settings x 4 seeds
    "q12": dict(n=12, n_train=40, n_test=20, seeds=range(4), fmaps=["zz"],
                datasets=["digits3v5", "qnative"],
                noises=["ideal", "dep", "coh"]),
}


def main():
    arm = sys.argv[1]
    a = ARMS[arm]
    only = sys.argv[2:]  # optional filters, e.g. noise names
    for seed in a["seeds"]:
        for dataset in a["datasets"]:
            for fmap in a["fmaps"]:
                for noise in a["noises"]:
                    if only and noise not in only:
                        continue
                    run_config(arm, fmap, a["n"], dataset, seed, noise,
                               a["n_train"], a["n_test"])


if __name__ == "__main__":
    main()
