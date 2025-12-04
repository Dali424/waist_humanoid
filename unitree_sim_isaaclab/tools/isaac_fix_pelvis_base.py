#!/usr/bin/env python3
"""
Isaac Sim utility: ensure a single articulation under a robot subtree and fix the pelvis to world.

Two supported methods:
- fixed_joint (default): create a PhysicsFixedJoint that anchors pelvis to world (no body0).
- fixed_base: set physxArticulation:fixedBase on the articulation root prim.

Usage examples (inside Isaac Sim Python or Kit headless):

  # Inside Script Editor (stage already loaded):
  import tools.isaac_fix_pelvis_base as fix
  fix.ensure_fixed_pelvis(robot_prim_path="/World/envs/env_0/Robot",
                          base_prim_path=None,
                          method="fixed_joint",
                          remove_other_articulations=True)

  # From command line (headless):
  # kit.exe --enable omni.kit.usd.layers --/app/quitAfter=1 --/script tools/isaac_fix_pelvis_base.py \
  #    --stage /path/to/scene.usd --robot-prim /World/envs/env_0/Robot --method fixed_joint --save /tmp/out.usd

This script assumes that the pelvis prim is at <robot_prim>/pelvis unless --base-prim is provided.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

try:
    import omni.usd  # Isaac Sim context
except Exception:
    omni = None  # type: ignore

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema
except Exception:  # pragma: no cover - PhysxSchema only exists in Isaac Sim
    PhysxSchema = None  # type: ignore


def _get_stage(stage_path: Optional[str]) -> Usd.Stage:
    if stage_path:
        # If running in a Kit context, use omni.usd to open; else open with Usd.Stage.Open
        if omni is not None:
            ctx = omni.usd.get_context()
            ctx.open_stage(stage_path)
            stage = ctx.get_stage()
        else:
            stage = Usd.Stage.Open(stage_path)
        if stage is None:
            raise RuntimeError(f"Failed to open stage: {stage_path}")
        return stage
    # Otherwise, get the currently open stage in Isaac Sim
    if omni is None:
        raise RuntimeError("No stage path provided and omni.usd not available.")
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No active stage found in Isaac Sim context.")
    return stage


def _list_articulation_roots(stage: Usd.Stage, root_prim: Usd.Prim) -> List[Usd.Prim]:
    prims: List[Usd.Prim] = []
    for p in stage.Traverse(root_prim.GetPath()):
        try:
            if UsdPhysics.ArticulationRootAPI(p):
                # HasAPI check is robust via schema instance validity
                if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prims.append(p)
                    continue
        except Exception:
            pass
        if PhysxSchema is not None:
            try:
                if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    prims.append(p)
            except Exception:
                pass
    # Deduplicate while preserving order
    uniq = []
    seen = set()
    for p in prims:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _remove_articulation_apis(prim: Usd.Prim) -> None:
    # Remove PhysicsArticulationRootAPI
    try:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    except Exception:
        # Best-effort fallback: ignore if API cannot be removed via method above
        pass
    # Remove PhysxArticulationAPI
    if PhysxSchema is not None:
        try:
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        except Exception:
            pass


def _ensure_articulation_on(prim: Usd.Prim) -> None:
    # Apply PhysicsArticulationRootAPI and PhysxArticulationAPI if not present
    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    if PhysxSchema is not None and not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(prim)


def _ensure_rigidbody(prim: Usd.Prim) -> None:
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)


def _create_or_update_fixed_joint_to_world(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    body1_prim: Usd.Prim,
) -> UsdPhysics.FixedJoint:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    # Ensure joint itself does NOT carry articulation root APIs
    _remove_articulation_apis(stage.GetPrimAtPath(joint_path))
    # Set body1 as pelvis; leave body0 unset to anchor to world
    body1_rel = joint.CreateBody1Rel()
    body1_rel.SetTargets([body1_prim.GetPath()])
    return joint


def ensure_fixed_pelvis(
    robot_prim_path: str,
    base_prim_path: Optional[str] = None,
    method: str = "fixed_joint",
    remove_other_articulations: bool = True,
    stage_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Ensure single articulation under robot prim and fix pelvis to world.

    Args:
        robot_prim_path: Prim path of the robot subtree (e.g., /World/envs/env_0/Robot)
        base_prim_path: Prim path of the pelvis; defaults to <robot_prim_path>/pelvis
        method: 'fixed_joint' (default) or 'fixed_base'
        remove_other_articulations: If True, remove articulation APIs from all other prims
        stage_path: If provided, open this stage. Otherwise operate on current stage
        save_path: If provided, save the stage to this path after modification
    """
    stage = _get_stage(stage_path)

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim not found: {robot_prim_path}")

    pelvis_path = base_prim_path or (robot_prim_path.rstrip("/") + "/pelvis")
    pelvis_prim = stage.GetPrimAtPath(pelvis_path)
    if not pelvis_prim or not pelvis_prim.IsValid():
        raise RuntimeError(f"Pelvis prim not found: {pelvis_path}")

    # Ensure pelvis is a rigidbody
    _ensure_rigidbody(pelvis_prim)

    # Find articulation roots beneath robot
    art_roots = _list_articulation_roots(stage, robot_prim)

    # Keep articulation on pelvis; remove from others if requested
    if remove_other_articulations:
        for p in art_roots:
            if p.GetPath() != pelvis_prim.GetPath():
                _remove_articulation_apis(p)

    # Ensure articulation exists on pelvis (single root)
    _ensure_articulation_on(pelvis_prim)

    # Apply chosen fixing method
    method = method.lower().strip()
    if method == "fixed_joint":
        joint_path = Sdf.Path(robot_prim_path.rstrip("/") + "/root_fixed_joint")
        _create_or_update_fixed_joint_to_world(stage, joint_path, pelvis_prim)
    elif method == "fixed_base":
        if PhysxSchema is None:
            raise RuntimeError("PhysxSchema not available; 'fixed_base' method requires Isaac Sim.")
        physx_art = PhysxSchema.PhysxArticulationAPI(pelvis_prim)
        if not physx_art:
            physx_art = PhysxSchema.PhysxArticulationAPI.Apply(pelvis_prim)
        # Set physxArticulation:fixedBase = true
        try:
            physx_art.CreateFixedBaseAttr().Set(True)
        except Exception:
            # Older schema names fallback
            try:
                physx_art.GetFixedBaseAttr().Set(True)  # type: ignore
            except Exception as e:
                raise
    else:
        raise ValueError("method must be 'fixed_joint' or 'fixed_base'")

    # Optional: save
    if save_path:
        if omni is not None:
            ctx = omni.usd.get_context()
            ctx.save_as_stage(save_path)
        else:
            stage.Export(save_path)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix pelvis to world and ensure single articulation root")
    p.add_argument("--stage", type=str, default=None, help="Path to stage USD to open")
    p.add_argument("--robot-prim", type=str, required=True, help="Robot prim path, e.g., /World/envs/env_0/Robot")
    p.add_argument("--base-prim", type=str, default=None, help="Pelvis prim path (defaults to <robot>/pelvis)")
    p.add_argument("--method", type=str, default="fixed_joint", choices=["fixed_joint", "fixed_base"], help="Fixing method")
    p.add_argument("--keep-others", action="store_true", help="Do not remove articulation APIs from other prims")
    p.add_argument("--save", type=str, default=None, help="Path to save the modified stage")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    ensure_fixed_pelvis(
        robot_prim_path=ns.robot_prim,
        base_prim_path=ns.base_prim,
        method=ns.method,
        remove_other_articulations=not ns.keep_others,
        stage_path=ns.stage,
        save_path=ns.save,
    )


if __name__ == "__main__":
    main()

