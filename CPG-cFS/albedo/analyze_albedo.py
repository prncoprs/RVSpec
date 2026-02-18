#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, subprocess, sys, math, re, csv, shutil
from pathlib import Path
import numpy as np

def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{r.stderr.strip()}")
    return r.stdout

# ---------- robust numeric readers ----------
_FLOAT_RE = re.compile(r'(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|NAN)')

def read_numeric_cols(path, need=0, offset=0):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            toks = _FLOAT_RE.findall(ln)
            if need == 0:
                if not toks:
                    continue
                vals = [float(t) if t.lower()!='nan' else math.nan for t in toks]
                rows.append(vals)
            else:
                if len(toks) < offset + need:
                    continue
                sel = toks[offset:offset+need]
                vals = [float(t) if t.lower()!='nan' else math.nan for t in sel]
                rows.append(vals)
    return rows

def read_vector3_first3(path):
    return read_numeric_cols(path, need=3, offset=0)

# ---------- math helpers ----------
def unit(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else [0.0,0.0,0.0]

def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def quat_to_cbn(q):
    # qbn = [q1 q2 q3 q4], q4 scalar; returns C_BN (B<-N)
    q1,q2,q3,q4 = q
    n = math.sqrt(q1*q1+q2*q2+q3*q3+q4*q4)
    if n == 0: return [[1,0,0],[0,1,0],[0,0,1]]
    q1,q2,q3,q4 = q1/n, q2/n, q3/n, q4/n
    C = [[0.0]*3 for _ in range(3)]
    C[0][0] = 1 - 2*(q2*q2 + q3*q3)
    C[0][1] = 2*(q1*q2 - q4*q3)
    C[0][2] = 2*(q1*q3 + q4*q2)
    C[1][0] = 2*(q1*q2 + q4*q3)
    C[1][1] = 1 - 2*(q1*q1 + q3*q3)
    C[1][2] = 2*(q2*q3 - q4*q1)
    C[2][0] = 2*(q1*q3 - q4*q2)
    C[2][1] = 2*(q2*q3 + q4*q1)
    C[2][2] = 1 - 2*(q1*q1 + q2*q2)
    return C

def matvec(C, v):
    return [C[0][0]*v[0]+C[0][1]*v[1]+C[0][2]*v[2],
            C[1][0]*v[0]+C[1][1]*v[1]+C[1][2]*v[2],
            C[2][0]*v[0]+C[2][1]*v[1]+C[2][2]*v[2]]

def angle_deg(a,b):
    aa, bb = unit(a), unit(b)
    c = max(-1.0, min(1.0, dot(aa,bb)))
    return math.degrees(math.acos(c))

# ---------- parse helpers ----------
def detect_sc_file_from_inp_sim(inp_sim_path):
    """
     Inp_Sim.txt Spacecraft  SC  token SC_NOS3.txt
    TRUE  0 SC_NOS3.txt
    """
    sc_file = None
    with open(inp_sim_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("<"):
                continue
            toks = ln.split()
            if len(toks) >= 3 and toks[0] in ("TRUE","FALSE") and toks[2].endswith(".txt"):
                sc_file = toks[2]
                break
    return sc_file

def parse_css_from_sc_file(sc_path):
    """
     SC_NOS3.txt  SC_SensorFOV.txt  CSS 
    
       +1: Sample Time
       +2: Axis (3 floats)
       +3: Half-cone Angle (deg)
       +4: Scale
       +5: Quantization
       +6: Body
       +7: Node
    """
    axes, fov_deg = [], []
    lines = Path(sc_path).read_text().splitlines()
    i = 0
    Ncss_declared = None
    #  "Number of Coarse Sun Sensors"
    for j,ln in enumerate(lines):
        if "Coarse Sun Sensor" in ln:
            for k in range(j+1, min(j+6, len(lines))):
                m = re.search(r'^\s*(\d+)\s*!?\s*Number of Coarse Sun Sensors', lines[k])
                if m:
                    Ncss_declared = int(m.group(1))
                    break
            break

    # 
    while i < len(lines):
        if lines[i].strip().startswith("============================== CSS"):
            if i+3 < len(lines):
                try:
                    ax = [float(x) for x in re.split(r"\s+", lines[i+2].strip())[:3]]
                    half = float(re.split(r"\s+", lines[i+3].strip())[0])
                    axes.append(unit(ax))
                    fov_deg.append(half)
                except Exception:
                    pass
            i += 4
        else:
            i += 1

    # 
    if not axes and Ncss_declared:
        base = None
        for j,ln in enumerate(lines):
            if re.search(r'Number of Coarse Sun Sensors', ln):
                base = j + 1
                break
        if base is not None:
            i = base
            for _ in range(Ncss_declared):
                if i+2 < len(lines):
                    try:
                        ax = [float(x) for x in re.split(r"\s+", lines[i+1].strip())[:3]]
                        half = float(re.split(r"\s+", lines[i+2].strip())[0])
                        axes.append(unit(ax))
                        fov_deg.append(half)
                    except Exception:
                        pass
                i += 7
    return axes, fov_deg

def count_cols_first_nonempty(path):
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = _FLOAT_RE.findall(ln)
            if toks:
                return len(toks)
    return 0

def atomic_copy_from_container(container, container_path, dest_dir):
    """
     docker cp  time.42  Steps=0
    """
    snap = "/tmp/NOS3InOut_snap.tgz"
    cmd_pack = [
        "docker","exec",container,"bash","-lc",
        f"cd {container_path} && tar -czf {snap} ."
    ]
    run(cmd_pack)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_tgz = dest_dir/"NOS3InOut_snap.tgz"
    run(["docker","cp",f"{container}:{snap}",str(local_tgz)])
    # 
    target = dest_dir/"NOS3InOut"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    run(["tar","-xzf",str(local_tgz),"-C",str(target)])
    return target

# ---------- CSS-based reconstruction ----------
def css_reconstruct_direction(intensities, axesB, fov_deg, nadirB):
    """
    “” CSS 
    -  FOV  CSS 
    - 
    -  FOV >0  None 
    """
    wsum = [0.0, 0.0, 0.0]
    total = 0.0
    for I, axB, hf in zip(intensities, axesB, fov_deg):
        #  nadirB  CSS 
        th = angle_deg(axB, nadirB)
        if th <= hf + 1e-9 and I > 0.0:
            wsum[0] += I * axB[0]
            wsum[1] += I * axB[1]
            wsum[2] += I * axB[2]
            total += I
    if total <= 0.0:
        return None
    return unit(wsum)

def angle_stats(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, 0)
    return (float(a.mean()), float(np.median(a)), float(np.percentile(a,95)), float(a.max()), int(a.size))

def masked(arr, mask):
    a = np.array(arr, dtype=float)
    m = np.array(mask, dtype=bool)
    return a[m].tolist()

# ============================== MAIN ==============================

def main():
    ap = argparse.ArgumentParser(description="Copy NOS3InOut atomically and analyze CSS albedo pointing (global/daylight + self-check).")
    ap.add_argument("--container", default="sc01-fortytwo")
    ap.add_argument("--container-path", default="<HOME>/.nos3/42/NOS3InOut")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--local-dir", help="Analyze from a local NOS3InOut directory (skip docker cp).")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # 1)  local-dir
    if args.local_dir:
        local_nos3 = Path(args.local_dir).expanduser().resolve()
        if not local_nos3.exists():
            sys.exit(f"--local-dir not found: {local_nos3}")
        print(f"[1/5] Using local dir: {local_nos3}")
    else:
        print(f"[1/5] Snapshot-copy from {args.container}:{args.container_path} -> {dest}")
        local_nos3 = atomic_copy_from_container(args.container, args.container_path, dest)
        print(f"      Local path: {local_nos3}")

    # 2) load files
    time_path = local_nos3/"time.42"
    pos_path  = local_nos3/"PosN.42"
    qbn_path  = local_nos3/"qbn.42"
    svn_path  = local_nos3/"svn.42"
    svb_path  = local_nos3/"svb.42"
    alb_path  = local_nos3/"Albedo.42"
    ill_path  = local_nos3/"Illum.42"
    inp_sim   = local_nos3/"Inp_Sim.txt"

    for p in [pos_path,qbn_path,svn_path,svb_path,alb_path,ill_path,inp_sim]:
        if not p.exists():
            sys.exit(f"Missing file: {p}")

    #  SC 
    sc_file_name = detect_sc_file_from_inp_sim(inp_sim)
    sc_path = (local_nos3/sc_file_name) if sc_file_name else (local_nos3/"SC_NOS3.txt")
    if not sc_path.exists():
        sc_path = local_nos3/"SC_SensorFOV.txt"  # 
    if not sc_path.exists():
        print(f"[WARN] Cannot find SC file with CSS definitions. Will fall back to default ±X/±Y/±Z.")
        sc_path = None

    #  Ncss
    ncss_from_alb = count_cols_first_nonempty(alb_path)
    ncss_from_ill = count_cols_first_nonempty(ill_path)
    ncss = ncss_from_alb if ncss_from_alb>0 else ncss_from_ill
    if ncss <= 0:
        sys.exit("Cannot detect Ncss from Albedo.42 or Illum.42")

    #  CSS 
    axesB, fov_deg = [], []
    if sc_path:
        axesB, fov_deg = parse_css_from_sc_file(sc_path)
    if not axesB or len(axesB) != ncss:
        print(f"[WARN] CSS count mismatch: parsed={len(axesB)} vs data={ncss}. Aligning by repetition / truncation.")
        default_axes = [
            [ 1, 0, 0], [-1, 0, 0],
            [ 0, 1, 0], [ 0,-1, 0],
            [ 0, 0, 1], [ 0, 0,-1],
        ]
        axesB = ([unit(a) for a in default_axes] * ((ncss+5)//6))[:ncss]
        fov_deg = [90.0]*ncss

    # 
    times = [ln.strip() for ln in open(time_path)] if time_path.exists() else []
    posN  = read_vector3_first3(pos_path)
    qbn   = read_numeric_cols(qbn_path, need=4, offset=0)
    svn   = read_numeric_cols(svn_path, need=3, offset=0)
    svb   = read_numeric_cols(svb_path, need=3, offset=0)
    alb   = read_numeric_cols(alb_path, need=0)
    ill   = read_numeric_cols(ill_path, need=0)

    # 
    candidates = [len(posN),len(qbn),len(svn),len(svb),len(alb),len(ill)]
    if times:
        candidates.append(len(times))
    nsteps = min([c for c in candidates if c>0]) if candidates else 0
    if nsteps == 0:
        sys.exit("No data rows found (nsteps=0). Check snapshot/copy.")

    # 
    daylight_mask = []
    for k in range(nsteps):
        rhatN = unit(posN[k])
        sN    = unit(svn[k])
        daylight_mask.append(1 if dot(sN, rhatN) > 0 else 0)

    print("[2/5] Dataset")
    print(f"- Steps: {nsteps}")
    print(f"- Daylight frames: {sum(daylight_mask)}/{nsteps}")

    # 3)  & 
    err_with, err_no, err_syn = [], [], []
    err_with_day, err_no_day, err_syn_day = [], [], []

    selfcheck_angles = []  # svb vs (CBN*svn)
    selfcheck_angles_day = []

    valid_with = valid_no = valid_syn = 0
    valid_with_day = valid_no_day = valid_syn_day = 0

    out_csv = dest/"albedo_geometry_summary.csv"
    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f)
        header=["step_idx","daylight","with_alb_sum","no_alb_sum"] + \
                [f"ill_css{i}" for i in range(ncss)] + \
                [f"alb_css{i}" for i in range(ncss)]
        w.writerow(header)

        for k in range(nsteps):
            rhatN = unit(posN[k])
            CBN   = quat_to_cbn(qbn[k])
            nadirB= matvec(CBN,[-rhatN[0],-rhatN[1],-rhatN[2]])
            sN    = unit(svn[k])
            sB_truth = unit(svb[k])
            is_day = daylight_mask[k]==1

            # 
            calc_sB = unit(matvec(CBN, sN))
            ang_chk = angle_deg(sB_truth, calc_sB)
            selfcheck_angles.append(ang_chk)
            if is_day: selfcheck_angles_day.append(ang_chk)

            # 
            ill_row = ill[k][:ncss] + [0.0]*max(0, ncss-len(ill[k]))
            alb_row = alb[k][:ncss] + [0.0]*max(0, ncss-len(alb[k]))

            with_vec = ill_row
            no_vec   = [max(0.0, ill_row[i]-alb_row[i]) for i in range(ncss)]
            syn_vec  = no_vec[:]  #  no_alb 

            # 
            sB_with = css_reconstruct_direction(with_vec, axesB, fov_deg, nadirB)
            sB_no   = css_reconstruct_direction(no_vec,   axesB, fov_deg, nadirB)
            sB_syn  = css_reconstruct_direction(syn_vec,  axesB, fov_deg, nadirB)

            # 
            if sB_with is not None:
                valid_with += 1
                err_with.append(angle_deg(sB_with, sB_truth))
                if is_day:
                    valid_with_day += 1
                    err_with_day.append(angle_deg(sB_with, sB_truth))

            if sB_no is not None:
                valid_no += 1
                err_no.append(angle_deg(sB_no, sB_truth))
                if is_day:
                    valid_no_day += 1
                    err_no_day.append(angle_deg(sB_no, sB_truth))

            if sB_syn is not None:
                valid_syn += 1
                err_syn.append(angle_deg(sB_syn, sB_truth))
                if is_day:
                    valid_syn_day += 1
                    err_syn_day.append(angle_deg(sB_syn, sB_truth))

            w.writerow([k+1, int(is_day), sum(with_vec), sum(no_vec)] +
                       [f"{v:.6g}" for v in ill_row] + [f"{v:.6g}" for v in alb_row])

    # 4) 
    print("[3/5] Availability")
    print(f"- Valid recon (with_alb): {valid_with}/{nsteps}")
    print(f"- Valid recon (no_alb):   {valid_no}/{nsteps}")
    print(f"- Valid recon (synth_dir):{valid_syn}/{nsteps}")

    def print_block(tag, e_with, e_no, e_syn, Nw, Nn, Ns):
        m1, md1, p951, mx1, _ = angle_stats(e_with)
        m2, md2, p952, mx2, _ = angle_stats(e_no)
        m3, md3, p953, mx3, _ = angle_stats(e_syn)
        # with - no
        if len(e_with)==len(e_no) and len(e_with)>0:
            delta = (np.array(e_with)-np.array(e_no)).tolist()
            dm, dmed, dp95, dmax, dN = angle_stats(delta)
            delta_str = f"  delta(with−no): mean={dm:.3f}, median={dmed:.3f}, p95={dp95:.3f}, max={dmax:.3f}  (N={dN})"
        else:
            delta_str = "  delta(with−no): N/A (different availability)"
        print(f"[4/5] Pointing error vs truth (deg) - {tag}")
        print(f"  ({tag.lower()})")
        print(f"  with_alb : mean={m1:.3f}, median={md1:.3f}, p95={p951:.3f}, max={mx1:.3f}  (N={Nw})")
        print(f"  no_alb   : mean={m2:.3f}, median={md2:.3f}, p95={p952:.3f}, max={mx2:.3f}  (N={Nn})")
        print(f"  synth_dir: mean={m3:.3f}, median={md3:.3f}, p95={p953:.3f}, max={mx3:.3f}  (N={Ns})")
        print(delta_str)

    print_block("GLOBAL (all frames)", err_with, err_no, err_syn,
                len(err_with), len(err_no), len(err_syn))
    print_block("DAYLIGHT ONLY", err_with_day, err_no_day, err_syn_day,
                len(err_with_day), len(err_no_day), len(err_syn_day))

    # 
    g_mean,g_med,g_p95,g_max,g_N = angle_stats(selfcheck_angles)
    d_mean,d_med,d_p95,d_max,d_N = angle_stats(selfcheck_angles_day)
    print("\n[5/5] Self-check angle: angle( svb , CBN*svn )  [deg]")
    print(f"  global  : mean={g_mean:.3f}, median={g_med:.3f}, p95={g_p95:.3f}, max={g_max:.3f}  (N={g_N})")
    print(f"  daylight: mean={d_mean:.3f}, median={d_med:.3f}, p95={d_p95:.3f}, max={d_max:.3f}  (N={d_N})")

    print(f"\nCSV written: {out_csv}")
    print("Notes:")
    print("- with_alb  = Illum.42")
    print("- no_alb    = max(0, Illum.42 − Albedo.42)")
    print("- synth_dir =  no_alb  no_alb ")
    print("-  svb.42Bdaylight  dot(sun_N, rhat_N) > 0")

if __name__=="__main__":
    main()
