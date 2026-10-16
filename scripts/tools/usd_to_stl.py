#!/usr/bin/env python3
"""Convert USD prop/asset meshes to STL for 3D printing or external tooling.

Bakes each mesh's world transform (including any scale) into the output
vertices, and converts USD stage units to meters by default. STL does not
store units: import the result as meters, or use --units mm for 3D printing.

Usage:
    python scripts/tools/usd_to_stl.py assets/props/light_blue_cube.usd
    python scripts/tools/usd_to_stl.py assets/props/*.usd --output assets/props_stl
    python scripts/tools/usd_to_stl.py assets/props --output assets/props_stl
    python scripts/tools/usd_to_stl.py assets/props --output assets/props_stl_mm --units mm
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom


def _fan_triangulate(indices: list[int]) -> list[tuple[int, int, int]]:
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def _mesh_prim_to_trimesh(prim: Usd.Prim, meters_per_unit: float, units: str = "m") -> trimesh.Trimesh:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    output_scale = meters_per_unit * {"m": 1.0, "mm": 1000.0}[units]
    world_points = [xf.Transform(Gf.Vec3d(*p)) * output_scale for p in points]
    vertices = np.array(world_points, dtype=np.float64)

    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    cursor = 0
    for count in face_counts:
        faces.extend(_fan_triangulate(face_indices[cursor : cursor + count]))
        cursor += count

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def convert_usd_to_stl(usd_path: Path, output_dir: Path, units: str = "m") -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"  [skip] could not open {usd_path}")
        return

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    meshes = [
        _mesh_prim_to_trimesh(prim, meters_per_unit, units)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    ]
    if not meshes:
        print(f"  [skip] no mesh found in {usd_path}")
        return

    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (usd_path.stem + ".stl")
    combined.export(str(out_path))

    size = combined.extents  # In the selected output units.
    print(
        f"  {usd_path.name} -> {out_path}  "
        f"({len(combined.vertices)} verts, {len(combined.faces)} tris, "
        f"size {size[0]:.6g}x{size[1]:.6g}x{size[2]:.6g} {units})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="+", help="USD file(s) and/or directories containing .usd files")
    parser.add_argument("--output", default="assets/props_stl", help="Output directory for .stl files")
    parser.add_argument("--units", choices=("m", "mm"), default="m",
                        help="STL coordinate units (default: m; use mm for 3D printing)")
    args = parser.parse_args()

    usd_paths: list[Path] = []
    for entry in args.input:
        p = Path(entry)
        if p.is_dir():
            usd_paths.extend(sorted(p.glob("*.usd")))
        elif p.exists():
            usd_paths.append(p)
        else:
            print(f"  [skip] not found: {entry}")

    output_dir = Path(args.output)
    print(f"Converting {len(usd_paths)} USD file(s) -> {output_dir}/")
    for usd_path in usd_paths:
        convert_usd_to_stl(usd_path, output_dir, args.units)

    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
