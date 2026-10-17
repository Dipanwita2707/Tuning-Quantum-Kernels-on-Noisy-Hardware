"""Cross-vendor diagnostic check via Open Quantum (second account).

Measures, on a non-IBM QPU, the entries the label-free diagnostics need:
forward K(x_i, x_j), swapped K(x_j, x_i) for all pairs of N points, and
the diagonal K(x_i, x_i), for 4-qubit ZZ kernels. Several 4-qubit blocks
are packed into one circuit on contiguous virtual qubits, with one extra
qubit in |1> to fix the bit order of returned strings.

Run with .venv-oq. Every job is quoted first; submission stops before
total spend would exceed --cap credits.

  python oq_diag.py --quote --backend iqm:garnet
  python oq_diag.py --submit --backend iqm:garnet --cap 20
  python oq_diag.py --collect --backend iqm:garnet
"""
import argparse
import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

import openquantum_hw as oq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
N_Q, SHOTS, SEED, C = 4, 500, 0, 0.25
BLOCKS = {"iqm:garnet": 4, "iqm:emerald": 12, "ionq:forte-1": 8}
N_PTS_BY = {"iqm:garnet": 6, "iqm:emerald": 8, "ionq:forte-1": 6}
N_PTS = 8


def points():
    """First N_PTS training points of the 4-qubit digits 3-vs-5 task, with
    the same preprocessing as ext_engine.load (seed 0)."""
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    rng = np.random.RandomState(SEED)
    d = load_digits()
    m = (d.target == 3) | (d.target == 5)
    X, y = d.data[m], (d.target[m] == 5).astype(int)
    idx = rng.choice(len(X), 120, replace=False)
    X, y = X[idx], y[idx]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=40, random_state=SEED,
                                      stratify=y)
    Xtr = Xtr[:80]
    Xtr = PCA(n_components=N_Q, random_state=SEED).fit(Xtr).transform(Xtr)
    Xtr = MinMaxScaler((0, np.pi)).fit(Xtr).transform(Xtr)
    return Xtr[:N_PTS]


def entries():
    X = points()
    fm = zz_feature_map(N_Q, reps=2, entanglement="linear")
    pairs = [(i, j) for i in range(N_PTS) for j in range(i + 1, N_PTS)]
    # "rep" re-measures every forward circuit in a different (random)
    # block: forward-vs-repeat spread measures block-to-block (qubit)
    # variation, so only asymmetry beyond it is attributed to direction
    spec = ([("fwd", i, j) for i, j in pairs] +
            [("swap", j, i) for i, j in pairs] +
            [("rep", i, j) for i, j in pairs] +
            [("diag", i, i) for i in range(N_PTS)])
    out = []
    for kind, a, b in spec:
        qc = QuantumCircuit(N_Q)
        qc.compose(fm.assign_parameters(C * X[a]), inplace=True)
        # the barrier stops the transpiler from cancelling U(x)^dag U(x)
        # on diagonal entries, so every circuit runs in full
        qc.barrier()
        qc.compose(fm.assign_parameters(C * X[b]).inverse(), inplace=True)
        exact = abs(Statevector(qc.remove_final_measurements(inplace=False)
                                if qc.num_clbits else qc).data[0]) ** 2
        qc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                       optimization_level=1, seed_transpiler=SEED)
        out.append({"kind": kind, "a": a, "b": b, "exact": float(exact),
                    "circ": qc})
    order = np.random.default_rng(SEED).permutation(len(out))
    return [out[k] for k in order]


def packed(chunk):
    width = N_Q * len(chunk) + 1
    qc = QuantumCircuit(width, width)
    for k, e in enumerate(chunk):
        qc.compose(e["circ"], qubits=range(N_Q * k, N_Q * (k + 1)),
                   inplace=True)
    qc.x(width - 1)
    for q in range(width):
        qc.measure(q, q)
    return qc


def decode(counts, n_blocks):
    width = N_Q * n_blocks + 1
    rows = [(oq.to_bits(k, width), v) for k, v in counts.items()]
    left = sum(v for s, v in rows if s[0] == "1")
    right = sum(v for s, v in rows if s[-1] == "1")
    tot = sum(v for _, v in rows)
    p0 = np.zeros(n_blocks)
    for s, v in rows:
        s = s[::-1] if left >= right else s
        for b in range(n_blocks):
            if "1" not in s[N_Q * b:N_Q * (b + 1)]:
                p0[b] += v
    return p0 / tot, max(left, right) / tot


def service2():
    from openquantum_sdk.auth import ClientCredentials
    from openquantum_sdk_qiskit import OpenQuantumService
    k = json.load(open(os.path.join(ROOT, ".local", "credentials", "apikey.json")))
    return OpenQuantumService(creds=ClientCredentials(
        client_id=k["oq_client_id2"], client_secret=k["oq_client_secret2"]))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--quote", "--submit", "--collect"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--backend", default="iqm:garnet")
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--shots", type=int, default=SHOTS)
    a = ap.parse_args()
    out = os.path.join(ROOT, "data", "extended", "oq_diag",
                       a.backend.replace(":", "_") + f"_s{a.shots}")
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    global N_PTS
    N_PTS = N_PTS_BY[a.backend]
    for attempt in range(6):
        try:
            svc = service2()
            backend = svc.return_backend(a.backend)
            sched = backend._scheduler
            break
        except Exception as e:
            if "401" not in str(e):
                raise
            time.sleep(20)
    oq.BACKEND_ID, oq.SHOTS = a.backend, a.shots
    ents = entries()
    B = BLOCKS[a.backend]
    chunks = [ents[k:k + B] for k in range(0, len(ents), B)]

    if a.quote:
        qc = packed(chunks[0])
        for attempt in range(6):
            try:
                _, _, _, cost = oq.prepare(backend, sched, qc, "quote-probe")
                break
            except Exception as e:
                if "401" not in str(e):
                    raise
                time.sleep(20)
                svc = service2()
                backend = svc.return_backend(a.backend)
                sched = backend._scheduler
        print(f"{a.backend}: {len(ents)} entries in {len(chunks)} jobs of "
              f"{B} blocks ({qc.num_qubits} qubits); one job = {cost} credits;"
              f" total ~{cost * len(chunks)}; balance {oq.balance(svc)}")
        return

    log_p = os.path.join(out, "jobs.json")
    log = json.load(open(log_p)) if os.path.exists(log_p) else []
    if a.submit:
        if a.cap is None:
            raise SystemExit("--cap is required")
        from openquantum_sdk.models import APIError, JobCreate
        spent = sum(r["cost"] for r in log)
        done = {r["chunk"] for r in log}
        for n, ch in enumerate(chunks):
            if n in done:
                continue
            qc = packed(ch)
            for attempt in range(8):
                try:
                    prep, plan, prio, cost = oq.prepare(
                        backend, sched, qc, f"diag-{n}")
                    if spent + cost > a.cap:
                        print(f"STOP: spent {spent}, next {cost}, cap {a.cap}")
                        return
                    job = sched.create_job(JobCreate(
                        job_preparation_id=prep, execution_plan_id=plan,
                        queue_priority_id=prio))
                    break
                except APIError as e:
                    if "401" not in str(e) and "JOB_LIMIT" not in str(e):
                        raise
                    time.sleep(30 * (attempt + 1))
                    svc = service2()
                    backend = svc.return_backend(a.backend)
                    sched = backend._scheduler
            else:
                raise SystemExit(f"chunk {n}: could not submit")
            spent += cost
            log.append({"chunk": n, "job_id": job.id, "cost": cost,
                        "entries": [{k: e[k] for k in ("kind", "a", "b",
                                                       "exact")} for e in ch]})
            json.dump(log, open(log_p, "w"), indent=1)
            print(f"chunk {n}: {job.id} cost {cost} total {spent}", flush=True)
        print("submitted; credits", spent)

    if a.collect:
        import requests
        rows, pending = [], 0
        for r in log:
            raw = os.path.join(out, "raw", f"chunk{r['chunk']:02d}.json")
            if not os.path.exists(raw):
                job = sched.get_job(r["job_id"])
                if job.status != "Completed":
                    pending += 1
                    print("chunk", r["chunk"], job.status)
                    continue
                resp = requests.get(job.output_data_url, timeout=60)
                resp.raise_for_status()
                json.dump({"job_id": r["job_id"], "counts": resp.json()},
                          open(raw, "w"))
            counts = json.load(open(raw))["counts"]
            p0, sent = decode(counts, len(r["entries"]))
            for e, p in zip(r["entries"], p0):
                rows.append({**e, "hw": float(p), "sentinel": sent})
        json.dump(rows, open(os.path.join(out, "entries.json"), "w"), indent=1)
        print("pending", pending, "collected entries", len(rows))


if __name__ == "__main__":
    main()
