"""Extended-study hardware runs (PROTOCOL_extended_study.md).

One "run" = one (device, day, task, mitigation) block: for each bandwidth,
the off-diagonal training block, the measured diagonal, and swapped
estimates for a fixed pair subset, so the label-free diagnostics (SNR_hat,
asymmetry index, diagonal heterogeneity) and the diagonal-restored
protocol can be evaluated on hardware.

Every bandwidth is one job; before each job the script sums the quantum
seconds already charged to this run directory and stops if the next job
could exceed --budget-seconds.

  python hardware_ext.py --dry-run --backend ibm_fez --task digits3v5
  python hardware_ext.py --run --backend ibm_fez --task digits3v5 \
      --tag day1 --budget-seconds 300 [--mitigate]
  python hardware_ext.py --collect --backend ibm_fez --task digits3v5 --tag day1
"""
import argparse
import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile

import ext_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "extended", "hardware")
N_TRAIN, N_SWAP, SHOTS, SEED = 32, 64, 512, 0
SEC_PER_CIRCUIT = 71 / 496   # measured on IBM Heron at 512 shots


def task_data(task, n):
    Xtr, _, ytr, _ = E.load(task, n, 80, 40, SEED)
    return Xtr[:N_TRAIN], ytr[:N_TRAIN]


def circuits(task, n, c):
    X, y = task_data(task, n)
    fm = E.feature_map("zz", n)
    N = len(X)
    iu = list(zip(*np.triu_indices(N, 1)))
    rng = np.random.default_rng(SEED)
    swap_idx = np.sort(rng.choice(len(iu), N_SWAP, replace=False))
    spec = ([("off", i, j) for i, j in iu] +
            [("diag", i, i) for i in range(N)] +
            [("swap", iu[k][1], iu[k][0]) for k in swap_idx])
    circs = []
    for _, a, b in spec:
        qc = QuantumCircuit(n)
        qc.compose(fm.assign_parameters(c * X[a]), inplace=True)
        # the barrier stops the transpiler from cancelling U(x)^dag U(x)
        # on diagonal entries, so every circuit runs in full
        qc.barrier()
        qc.compose(fm.assign_parameters(c * X[b]).inverse(), inplace=True)
        qc.measure_all()
        circs.append(qc)
    return circs, spec, y, swap_idx


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--collect", action="store_true")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--task", default="digits3v5")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--bandwidths", type=float, nargs="+",
                    default=[0.0625, 0.125, 0.25])
    ap.add_argument("--tag", default="day1")
    ap.add_argument("--mitigate", action="store_true",
                    help="Pauli twirling + dynamical decoupling")
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--shuffle", action="store_true",
                    help="randomize circuit order within each job (A4)")
    ap.add_argument("--account", default=None,
                    help="saved Qiskit account name (default: default account)")
    a = ap.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(name=a.account) if a.account else \
        QiskitRuntimeService()
    run_dir = os.path.join(OUT, f"{a.backend}_{a.task}_n{a.qubits}_{a.tag}"
                           + ("_mit" if a.mitigate else ""))
    os.makedirs(run_dir, exist_ok=True)
    log_p = os.path.join(run_dir, "jobs.json")
    log = json.load(open(log_p)) if os.path.exists(log_p) else {}

    if a.dry_run or a.run:
        if a.run and a.budget_seconds is None:
            raise SystemExit("--budget-seconds is required for --run")
        backend = svc.backend(a.backend)
        total = 0
        for c in a.bandwidths:
            if str(c) in log:
                continue
            circs, spec, _, _ = circuits(a.task, a.qubits, c)
            t = transpile(circs, backend, optimization_level=1,
                          seed_transpiler=SEED)
            if a.shuffle:
                # forward and swapped circuits interleave in time, so slow
                # drift within a job cannot mimic direction dependence
                perm = np.random.default_rng(SEED + int(1e4 * c)).permutation(
                    len(t))
                t = [t[k] for k in perm]
                np.save(os.path.join(run_dir, f"perm_c{c}.npy"), perm)
            est = len(t) * SEC_PER_CIRCUIT
            total += est
            if a.dry_run:
                print(f"c={c}: {len(t)} circuits, median depth "
                      f"{int(np.median([x.depth() for x in t]))}, median 2q "
                      f"{int(np.median([x.count_ops().get('cz', 0) for x in t]))}"
                      f", est. {est:.0f} s")
                continue
            used = sum(svc.job(v).metrics().get("usage", {})
                       .get("quantum_seconds", 0) for v in log.values())
            if used + 1.15 * est > a.budget_seconds:
                print(f"STOP before c={c}: used {used}s, next ~{est:.0f}s")
                break
            sampler = SamplerV2(mode=backend)
            if a.mitigate:
                sampler.options.twirling.enable_gates = True
                sampler.options.twirling.enable_measure = True
                sampler.options.dynamical_decoupling.enable = True
            job = sampler.run(t, shots=SHOTS)
            log[str(c)] = job.job_id()
            json.dump(log, open(log_p, "w"), indent=1)
            print(f"c={c}: submitted {job.job_id()} ({len(t)} circuits)",
                  flush=True)
            while str(job.status()) not in ("DONE", "ERROR", "CANCELLED"):
                time.sleep(20)
            print(f"c={c}: {job.status()}, "
                  f"{job.metrics().get('usage', {}).get('quantum_seconds')} s",
                  flush=True)
            if str(job.status()) != "DONE":
                break
        if a.dry_run:
            print(f"run total est. {total:.0f} s")

    if a.collect:
        for c in a.bandwidths:
            if str(c) not in log:
                continue
            job = svc.job(log[str(c)])
            if str(job.status()) != "DONE":
                print(f"c={c}: {job.status()}")
                continue
            _, spec, y, swap_idx = circuits(a.task, a.qubits, c)
            res = job.result()
            zero = "0" * a.qubits
            p = np.array([r.data.meas.get_counts().get(zero, 0) / SHOTS
                          for r in res])
            perm_p = os.path.join(run_dir, f"perm_c{c}.npy")
            if os.path.exists(perm_p):
                unshuf = np.empty_like(p)
                unshuf[np.load(perm_p)] = p
                p = unshuf
            kinds = np.array([s[0] for s in spec])
            m = job.metrics()
            np.savez_compressed(
                os.path.join(run_dir, f"gram_c{c}.npz"),
                off=p[kinds == "off"], diag=p[kinds == "diag"],
                swap=p[kinds == "swap"], swap_idx=swap_idx, y=y,
                quantum_seconds=m["usage"]["quantum_seconds"],
                finished=m["timestamps"]["finished"])
            print(f"c={c}: saved ({len(p)} values)")


if __name__ == "__main__":
    main()
