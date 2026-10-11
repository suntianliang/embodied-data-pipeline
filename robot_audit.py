# src/robot_audit.py
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

MJCF = "assets/unitree_g1/g1.xml"


def resolve_mjcf(path):
    """优先解析真实的模型 XML；若传的是 scene.xml，则追到被 include 的 g1.xml."""
    p = Path(path)
    if not p.exists():
        return str(p)

    try:
        root = ET.parse(str(p)).getroot()
    except ET.ParseError:
        return str(p)

    include = root.find("include")
    if include is not None:
        include_file = include.get("file")
        if include_file:
            included = (p.parent / include_file).resolve()
            if included.exists():
                return str(included)
    return str(p)


def walk_joints(node, parent=None):
    """递归遍历 MJCF body 树，收集所有 joint 及其限位"""
    rows = []
    for body in node.findall("body"):
        pname = body.get("name")
        for j in body.findall("joint"):
            name = j.get("name") or j.get("class") or "unnamed"
            jtype = j.get("type", "hinge")
            range_text = j.get("range", "-180 180")
            parts = range_text.split()
            lo = float(parts[0]) if parts else -180.0
            hi = float(parts[-1]) if parts else 180.0
            rows.append({
                "body": pname,
                "joint": name,
                "type": jtype,
                "limit_lo_deg": lo,
                "limit_hi_deg": hi,
                "span_deg": hi - lo,
                "parent": parent,
            })
        rows.extend(walk_joints(body, pname))
    return rows


def audit_mujoco():
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(MJCF)
        print(f"[OK] MuJoCo 加载成功: {MJCF}")
        print(f"     nq={m.nq}, nv={m.nv}, nbody={m.nbody}, njnt={m.njnt}")
        # 导出限位表
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        lo = m.jnt_range[:, 0]
        hi = m.jnt_range[:, 1]
        out = Path("reports/g1_joint_limits.csv")
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "name", "lo_deg", "hi_deg", "span_deg"])
            w.writeheader()
            for i, n in enumerate(names):
                w.writerow({"idx": i, "name": n, "lo_deg": round(lo[i], 2),
                            "hi_deg": round(hi[i], 2), "span_deg": round(hi[i] - lo[i], 2)})
        print(f"[OK] 限位表 -> {out.resolve()}")
    except Exception as e:
        print(f"[WARN] MuJoCo 加载失败（先不管，回头修）: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", default=MJCF)
    args = p.parse_args()
    mjcf_path = resolve_mjcf(args.mjcf)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    rows = walk_joints(worldbody if worldbody is not None else root)

    if not rows:
        raise RuntimeError(f"没有在 {mjcf_path} 中找到任何 joint；请确认 MJCF 是真正的机器人模型而不是 scene.xml")

    out = Path("reports/g1_joints_xml.csv")
    out.parent.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] XML 关节表 -> {out.resolve()}  共 {len(rows)} joints")
    audit_mujoco()