#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, subprocess, sys, math, re, csv, tempfile, shutil
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser(description="Copy NOS3InOut atomically and analyze CSS albedo geometry (SC_NOS3 aware).")
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
        print(f"[1/4] Using local dir: {local_nos3}")
    else:
        print(f"[1/4] Snapshot-copy from {args.container}:{args.container_path} -> {dest}")
        local_nos3 = atomic_copy_from_container(args.container, args.container_path, dest)
        print(f"      Local path: {local_nos3}")

    # 2) load files
    time_path = local_nos3/"time.42"
    pos_path  = local_nos3/"PosN.42"
    qbn_path  = local_nos3/"qbn.42"
    svn_path  = local_nos3/"svn.42"
    alb_path  = local_nos3/"Albedo.42"
    ill_path  = local_nos3/"Illum.42"
    inp_sim   = local_nos3/"Inp_Sim.txt"

    for p in [pos_path,qbn_path,svn_path,alb_path,ill_path,inp_sim]:
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
    ncss_data = ncss_from_alb if ncss_from_alb>0 else ncss_from_ill
    if ncss_data <= 0:
        sys.exit("Cannot detect Ncss from Albedo.42 or Illum.42")

    #  CSS 
    axesB, fov_deg = [], []
    if sc_path:
        axesB, fov_deg = parse_css_from_sc_file(sc_path)
    if not axesB or len(axesB) != ncss_data:
        print(f"[WARN] CSS parsed from {sc_path.name if sc_path else 'N/A'} = {len(axesB)} "
              f"but data shows Ncss={ncss_data}. Will align to data columns.")
        default_axes = [
            [ 1, 0, 0], [-1, 0, 0],
            [ 0, 1, 0], [ 0,-1, 0],
            [ 0, 0, 1], [ 0, 0,-1],
        ]
        axesB = ([unit(a) for a in default_axes] * ((ncss_data+5)//6))[:ncss_data]
        fov_deg = [90.0]*ncss_data

    # 
    times = [ln.strip() for ln in open(time_path)] if time_path.exists() else []
    posN  = read_vector3_first3(pos_path)
    qbn   = read_numeric_cols(qbn_path, need=4, offset=0)
    svn   = read_numeric_cols(svn_path, need=3, offset=0)
    alb   = read_numeric_cols(alb_path, need=0)
    ill   = read_numeric_cols(ill_path, need=0)

    #  time.42 
    candidates = [len(posN),len(qbn),len(svn),len(alb),len(ill)]
    if times:
        candidates.append(len(times))
    nsteps = min([c for c in candidates if c>0]) if candidates else 0
    if nsteps == 0:
        sys.exit("No data rows found (nsteps=0). Check snapshot/copy.")

    ncss   = ncss_data
    print(f"[2/4] Steps={nsteps}, CSS={ncss} (from data)")

    # 3) analyze
    out_csv = dest/"albedo_geometry_summary.csv"
    hit_frames=0
    first_nonzero_line=None
    first_nonzero_per_css=[None]*ncss
    nonzero_sum = 0.0

    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f)
        header=["step_idx","daylight","any_css_hit","hit_css_ids","min_angle_deg",
                "expected_nonzero","actual_albedo_sum"]+[f"alb_css{i}" for i in range(ncss)]
        w.writerow(header)

        for k in range(nsteps):
            rhatN=unit(posN[k])
            CBN=quat_to_cbn(qbn[k])
            nadirB=matvec(CBN,[-rhatN[0],-rhatN[1],-rhatN[2]])

            sN=unit(svn[k])
            daylight=(dot(sN,rhatN)>0)  # 

            hit_ids=[]
            minang=1e9
            for i,(axB,hf_deg) in enumerate(zip(axesB,fov_deg)):
                th=angle_deg(axB,nadirB)
                minang=min(minang,th)
                if th<=hf_deg+1e-9:
                    hit_ids.append(i)

            expected=daylight and bool(hit_ids)

            row = alb[k][:ncss] + [0.0]*max(0, ncss-len(alb[k]))
            alb_sum=sum(row); nonzero_sum += alb_sum

            if first_nonzero_line is None and alb_sum>0:
                first_nonzero_line=(k+1,row)
            for i in range(ncss):
                if first_nonzero_per_css[i] is None and row[i]>0:
                    first_nonzero_per_css[i]=(k+1,row[i])

            if expected: hit_frames+=1

            w.writerow([k+1,int(daylight),int(bool(hit_ids)),
                        ",".join(map(str,hit_ids)),f"{minang:.2f}",
                        int(expected),f"{alb_sum:.6g}"]+[f"{v:.6g}" for v in row])

    # 4) summary
    print("[3/4] Summary")
    print(f"- Frames with (daylight & earth-in-FOV): {hit_frames}/{nsteps}")
    if first_nonzero_line:
        ln,row=first_nonzero_line
        print(f"- First nonzero albedo: step {ln}, sum={sum(row):.6g}")
    else:
        print("- Albedo.42 all zeros")
    for i,info in enumerate(first_nonzero_per_css):
        if info:
            ln,val=info
            print(f"  CSS{i}: first nonzero @ step {ln}, val={val:.6g}")
        else:
            print(f"  CSS{i}: never nonzero")
    print(f"- Total albedo sum over range: {nonzero_sum:.6g}")
    print(f"[4/4] CSV written: {out_csv}")

    if hit_frames>0 and nonzero_sum==0.0:
        print("\n[NOTE]  Albedo  0 shader ")
        print("        time.42 ")

if __name__=="__main__":
    main()
