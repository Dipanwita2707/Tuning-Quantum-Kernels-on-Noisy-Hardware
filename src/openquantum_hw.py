"""Cross-platform hardware check via Open Quantum (second QPU vendor).

Runs the first N_SUB samples of the IBM hardware-validation set (same data,
same 8-qubit feature map, same bandwidths) on Rigetti Cepheus-1-108Q, so the
same Gram-matrix entries can be compared across IBM Heron, Rigetti, and
noiseless simulation.

Open Quantum bills every circuit as one task (1 credit each at our shot
count). To fit the budget, BLOCKS kernel circuits are executed in parallel
on disjoint 8-qubit chains of the 108-qubit device inside one circuit, and
each block's all-zeros frequency is read from its own 8 classical bits.
Kernel entries are assigned to blocks in a seeded random order so that
block (qubit-quality) effects are not confounded with entry structure. One
spare qubit is flipped to |1> and measured into the last classical bit, so
the bit order of the returned strings is determined from the data.

The SDK approves quotes automatically by default; this script instead
prepares each circuit, reads its quote, and refuses to submit anything that
would push total spend past --cap. Jobs are created asynchronously.

Run with the separate Qiskit-2 environment (.venv-oq):
  python openquantum_hw.py --list                 # backends + credit balance
  python openquantum_hw.py --quote                # price ONE circuit, no spend
  python openquantum_hw.py --submit --cap 22      # submit within cap (credits)
  python openquantum_hw.py --collect              # download finished results
Credentials: .local/credentials/apikey.json under the repository root,
any field names containing "client_id" (or a
value starting with "s_") and "secret".
"""
import argparse
import json
import os
import random
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import zz_feature_map
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "data", "preliminary", "hardware_oq")
PATHS = os.path.join(ROOT, "data", "preliminary", "hardware_oq_paths.json")
BACKEND_ID = "rigetti:cepheus-1-108q"
N_QUBITS, N_SUB, SHOTS, SEED = 8, 16, 1000, 42
BANDWIDTHS = [0.125, 0.5]
MAX_WIDTH = 107  # backend max_qubits_per_job


def load_subset():
    """Same preprocessing as common.load_dataset('digits', 8 features,
    n_samples=72, seed=42) used by hardware.py, reimplemented here because
    this environment runs Qiskit 2 (verified identical to 0.0 difference)."""
    rng = np.random.RandomState(SEED)
    d = load_digits()
    mask = (d.target == 3) | (d.target == 5)
    X, y = d.data[mask], (d.target[mask] == 5).astype(int)
    idx = rng.choice(len(X), size=72, replace=False)
    X, y = X[idx], y[idx]
    X = PCA(n_components=N_QUBITS, random_state=SEED).fit_transform(X)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                      stratify=y)
    Xtr = MinMaxScaler((0, np.pi)).fit(Xtr).transform(Xtr)
    return Xtr[:N_SUB], ytr[:N_SUB]


def kernel_block(x, z):
    """Unmeasured 8-qubit compute-uncompute circuit, lowered to plain gates
    with no layout (its CX gates act only on neighbours k, k+1)."""
    fm = zz_feature_map(N_QUBITS, reps=2, entanglement="linear")
    qc = QuantumCircuit(N_QUBITS)
    qc.compose(fm.assign_parameters(x), inplace=True)
    qc.compose(fm.assign_parameters(z).inverse(), inplace=True)
    return transpile(qc, basis_gates=["p", "cx", "h", "rz", "rx", "x"],
                     optimization_level=1, seed_transpiler=SEED)


def plan():
    """All (c, i, j) entries in seeded random order, chunked into packed
    circuits of len(paths) blocks each."""
    paths = json.load(open(PATHS))
    entries = [(c, i, j) for c in BANDWIDTHS
               for i in range(N_SUB) for j in range(i + 1, N_SUB)]
    random.Random(SEED).shuffle(entries)
    B = len(paths)
    chunks = [entries[k:k + B] for k in range(0, len(entries), B)]
    used = {q for p in paths for q in p}
    sentinel = min(q for q in range(MAX_WIDTH) if q not in used)
    return paths, chunks, sentinel


def packed_circuit(chunk, paths, sentinel, X):
    """Blocks on contiguous virtual qubits 8b..8b+7, sentinel last. The
    provider's compiler requires contiguous virtual qubits and performs
    placement and routing on the device itself; `paths` only fixes the
    number of blocks (12 disjoint 8-qubit chains exist on the device)."""
    B = len(paths)
    width = N_QUBITS * B + 1
    qc = QuantumCircuit(width, width)
    for b, (c, i, j) in enumerate(chunk):
        blk = kernel_block(c * X[i], c * X[j])
        qc.compose(blk, qubits=range(N_QUBITS * b, N_QUBITS * (b + 1)),
                   inplace=True)
    qc.x(width - 1)
    for b in range(B):
        for k in range(N_QUBITS):
            qc.measure(N_QUBITS * b + k, N_QUBITS * b + k)
    qc.measure(width - 1, width - 1)
    return qc


def service():
    from openquantum_sdk.auth import ClientCredentials
    from openquantum_sdk_qiskit import OpenQuantumService
    k = json.load(open(os.path.join(ROOT, ".local", "credentials", "apikey.json")))
    cid = next(v for n, v in k.items() if "client_id" in n.lower()
               or (isinstance(v, str) and v.startswith("s_")))
    sec = next(v for n, v in k.items() if "secret" in n.lower())
    return OpenQuantumService(creds=ClientCredentials(
        client_id=cid, client_secret=sec))


def balance(svc):
    mgmt = svc.management
    org = mgmt.list_user_organizations(limit=5).organizations[0].id
    return mgmt.get_credit_balance(org)


def prepare(backend, sched, qc, name):
    """Upload + prepare one circuit; return (prep_id, plan_id, prio_id, cost).
    Preparing a quote creates no job and spends nothing."""
    from openquantum_sdk.clients import JobSubmissionConfig
    from openquantum_sdk.models import JobPreparationCreate
    import qiskit.qasm3 as qasm3
    if qc.num_qubits > MAX_WIDTH:
        raise ValueError(f"circuit width {qc.num_qubits} > {MAX_WIDTH}")
    payload = qasm3.dumps(qc).encode("utf-8")
    cfg_data = backend.job_configuration_data(
        configuration_data={}, run_options={}, shots=SHOTS)
    cfg = JobSubmissionConfig(backend_class_id=BACKEND_ID, name=name,
                              job_subcategory_id="oth:oth",
                              configuration_data=cfg_data,
                              submitted_with="qiskit")
    org = sched._resolve_organization_id(None)
    upload_id = sched.upload_job_input(file_content=payload)
    shots, cfg_data = sched._normalize_shots_and_configuration(None, cfg_data)
    prep = sched.prepare_job(JobPreparationCreate(
        organization_id=org, backend_class_id=BACKEND_ID, name=name,
        upload_endpoint_id=upload_id, job_subcategory_id="oth:oth",
        shots=shots, configuration_data=cfg_data, submitted_with="qiskit",
        input_format=None))
    res = sched._wait_for_preparation(prep.id, timeout=300, interval=2.0)
    plan_id, prio_id = sched._choose_plan_and_priority(res, cfg)
    cost = sched._compute_total_cost(res.quote, plan_id, prio_id)
    return prep.id, plan_id, prio_id, cost


def to_bits(key, width):
    """Return the measured string in Qiskit order (clbit 0 rightmost)."""
    k = key.strip().replace(" ", "")
    if k.startswith("0x"):
        return format(int(k, 16), "b").zfill(width)
    return k.zfill(width)


def block_p0(counts, n_blocks):
    """Per-block all-zeros frequency, orienting bit order via the sentinel."""
    width = N_QUBITS * n_blocks + 1
    rows = [(to_bits(k, width), v) for k, v in counts.items()]
    left = sum(v for s, v in rows if s[0] == "1")
    right = sum(v for s, v in rows if s[-1] == "1")
    reverse = right > left  # sentinel rightmost => clbit 0 leftmost
    tot = sum(v for _, v in rows)
    p0 = np.zeros(n_blocks)
    for s, v in rows:
        s = s[::-1] if not reverse else s  # make index k == clbit k
        for b in range(n_blocks):
            if "1" not in s[N_QUBITS * b:N_QUBITS * (b + 1)]:
                p0[b] += v
    sentinel_ok = max(left, right) / tot
    return p0 / tot, reverse, sentinel_ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--list", "--quote", "--submit", "--collect"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--cap", type=float, default=None,
                    help="maximum total spend in credits (required to submit)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    svc = service()

    if a.list:
        for b in svc.backends():
            print(b)
        print("balance:", balance(svc))
        return

    backend = svc.return_backend(BACKEND_ID)
    sched = backend._scheduler
    from openquantum_sdk.models import JobCreate
    X, _ = load_subset()
    paths, chunks, sentinel = plan()

    if a.quote:
        qc = packed_circuit(chunks[0], paths, sentinel, X)
        print(f"packed circuit: width {qc.num_qubits}, depth {qc.depth()}, "
              f"{len(chunks[0])} blocks; {len(chunks)} circuits in total")
        _, _, _, cost = prepare(backend, sched, qc, "quote-probe")
        print(f"one circuit: {cost} credits -> total {cost * len(chunks)} "
              f"credits for {sum(map(len, chunks))} kernel entries")
        print("balance:", balance(svc))
        return

    if a.submit:
        if a.cap is None:
            raise SystemExit("--cap is required for --submit")
        log_path = os.path.join(OUT, "jobs.json")
        log = json.load(open(log_path)) if os.path.exists(log_path) else []
        spent = sum(r["cost"] for r in log)
        done = {r["chunk"] for r in log}
        from openquantum_sdk.models import APIError
        for n, chunk in enumerate(chunks):
            if n in done:
                continue
            qc = packed_circuit(chunk, paths, sentinel, X)
            # The service intermittently answers 401 to valid credentials.
            # A 401 means the request was rejected (nothing created, nothing
            # charged), so it is safe to log in again and retry; any other
            # error stops the run rather than risk a duplicate paid job.
            for attempt in range(8):
                try:
                    prep_id, plan_id, prio_id, cost = prepare(
                        backend, sched, qc, f"qk-packed-{n}")
                    if spent + cost > a.cap:
                        print(f"STOP: next circuit costs {cost}, spent "
                              f"{spent}, cap {a.cap}")
                        return
                    job = sched.create_job(JobCreate(
                        job_preparation_id=prep_id, execution_plan_id=plan_id,
                        queue_priority_id=prio_id))
                    break
                except APIError as e:
                    if "401" not in str(e):
                        raise
                    wait = 20 * (attempt + 1)
                    print(f"chunk {n}: 401, re-login in {wait}s", flush=True)
                    time.sleep(wait)
                    svc = service()
                    backend = svc.return_backend(BACKEND_ID)
                    sched = backend._scheduler
            else:
                raise SystemExit(f"chunk {n}: still unauthorized, stopping")
            spent += cost
            log.append({"chunk": n, "entries": chunk, "job_id": job.id,
                        "cost": cost})
            json.dump(log, open(log_path, "w"), indent=1)
            print(f"chunk {n}: job {job.id} cost {cost} total {spent}",
                  flush=True)
        print("all submitted; total credits", spent)
        return

    if a.collect:
        log = json.load(open(os.path.join(OUT, "jobs.json")))
        K = {c: np.eye(N_SUB) for c in BANDWIDTHS}
        filled = {c: np.eye(N_SUB, dtype=bool) for c in BANDWIDTHS}
        import requests
        raw_dir = os.path.join(OUT, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        pending, meta = 0, []
        for r in log:
            raw = os.path.join(raw_dir, f"chunk{r['chunk']:02d}.json")
            if not os.path.exists(raw):
                # the SDK's own download has no timeout and can hang forever
                counts = None
                for attempt in range(4):
                    try:
                        job = sched.get_job(r["job_id"])
                        if job.status != "Completed":
                            break
                        resp = requests.get(job.output_data_url, timeout=60)
                        resp.raise_for_status()
                        counts = resp.json()
                        break
                    except Exception as e:
                        print(f"chunk {r['chunk']}: {type(e).__name__}, "
                              f"retry {attempt + 1}", flush=True)
                        time.sleep(15)
                        svc = service()
                        sched = svc.return_backend(BACKEND_ID)._scheduler
                if counts is None:
                    pending += 1
                    print("chunk", r["chunk"], "not available yet", flush=True)
                    continue
                json.dump({"job_id": r["job_id"], "counts": counts},
                          open(raw, "w"))
            counts = json.load(open(raw))["counts"]
            p0, rev, ok = block_p0(counts, len(paths))
            meta.append({"chunk": r["chunk"], "reversed": rev,
                         "sentinel_one_frac": ok})
            for b, (c, i, j) in enumerate(r["entries"]):
                K[c][i, j] = K[c][j, i] = p0[b]
                filled[c][i, j] = filled[c][j, i] = True
        print("pending:", pending, "of", len(log))
        json.dump(meta, open(os.path.join(OUT, "orientation.json"), "w"),
                  indent=1)
        if pending == 0 and all(f.all() for f in filled.values()):
            _, y = load_subset()
            np.savez_compressed(os.path.join(OUT, "gram_oq.npz"), y=y,
                                **{f"K_c{c}": K[c] for c in BANDWIDTHS})
            print("saved", os.path.join(OUT, "gram_oq.npz"))


if __name__ == "__main__":
    main()
