import argparse, json, os
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from simpler_env.utils.env.env_builder import build_maniskill2_env

EGG_BOUNDS = None
O1 = "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
O2 = "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
TASKS = {
    "StackCube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", "bridge_table_1_v1", O1, "widowx",
                  (0.147, 0.028), (-0.26, -0.06, -0.10, 0.10), 0.07, [0, 1]),
    "Carrot": ("PutCarrotOnPlateInScene-v0", "bridge_table_1_v1", O1, "widowx",
               (0.147, 0.028), (-0.235, -0.085, -0.075, 0.075), 0.13, [0]),
    "Spoon": ("PutSpoonOnTableClothInScene-v0", "bridge_table_1_v1", O1, "widowx",
              (0.147, 0.028), (-0.235, -0.085, -0.075, 0.075), 0.13, [0]),
    "Eggplant": ("PutEggplantInBasketScene-v0", "bridge_table_1_v2", O2, "widowx_sink_camera_setup",
                 (0.127, 0.06), None, None, [0]),
}
MAX_DRIFT = {"StackCube": 0.01, "Carrot": 0.01, "Spoon": 0.01}


def propose(rng, task, xy_orig, quat_orig, orig_points):
    _, _, _, _, _, box, dmin, rotated = TASKS[task]
    xy = np.array(xy_orig, dtype=float).copy()
    if task == "Eggplant":
        lo, hi = EGG_BOUNDS[0], EGG_BOUNDS[1]
        xy[0] = [rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])]
    else:
        for i in range(2):
            xy[i] = [rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3])]
        if np.linalg.norm(xy[0] - xy[1]) < dmin:
            return None
        if min(np.min(np.linalg.norm(orig_points - xy[i], axis=1)) for i in range(2)) < 0.025:
            return None
    q = np.array(quat_orig, dtype=float).copy()
    for i in rotated:
        y0 = quat2euler(quat_orig[i])[2]
        rot_range = (15, 60) if task == "Eggplant" else (20, 45)
        q[i] = euler2quat(0, 0, y0 + np.radians(rng.uniform(*rot_range)) * rng.choice([-1, 1]))
    return xy, q


def check(env, task, xy, q):
    u = env.unwrapped
    saved = (u._xy_configs, u._quat_configs)
    u._xy_configs, u._quat_configs = [np.array(xy)], [np.array(q)]
    try:
        obs, _ = env.reset(options={
            "robot_init_options": {"init_xy": np.array(TASKS[task][4]), "init_rot_quat": np.array([0, 0, 0, 1])},
            "obj_init_options": {"episode_id": 0}})
    finally:
        u._xy_configs, u._quat_configs = saved
    settled = [np.array(p) for p in u.episode_obj_xyzs_after_settle]
    placed = np.array(u.obj_init_options["init_xys"])
    drift = [float(np.linalg.norm(settled[i][:2] - placed[i])) for i in range(len(settled))]
    z = [float(p[2]) for p in settled]
    cam = [c for c in obs["image"] if "Segmentation" in obs["image"][c]]
    visible = []
    if cam:
        seg = obs["image"][cam[-1]]["Segmentation"][..., 1]
        visible = [int((seg == o.id).sum()) for o in u.episode_objs]
    if task == "Eggplant":
        p = settled[0]
        lo, hi, z_lo, z_hi, px_min, orig_settled = EGG_BOUNDS
        ok = (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and z_lo <= p[2] <= z_hi
              and (not visible or visible[0] >= px_min)
              and np.min(np.linalg.norm(orig_settled[:, :2] - p[:2], axis=1)) >= 0.015)
        return ok, {"settled_xyz": np.round(p, 4).tolist(), "visible_px": visible}
    ok = (max(drift) <= MAX_DRIFT[task] and min(z) > u.scene_table_height - 0.02
          and (not visible or min(visible) >= 150))
    return ok, {"drift_m": [round(t, 4) for t in drift], "z": [round(v, 3) for v in z],
                "visible_px": visible}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="new scenes per task")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default="train_scenes.json")
    ap.add_argument("--tasks", default="StackCube,Carrot,Spoon,Eggplant")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    out, stats = {}, {}
    for task in a.tasks.split(","):
        env_name, scene, ov, robot, _, _, _, _ = TASKS[task]
        env = build_maniskill2_env(env_name, obs_mode="rgbd", robot=robot, sim_freq=500,
                                   control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                                   control_freq=5, max_episode_steps=120, scene_name=scene,
                                   camera_cfgs={"add_segmentation": True}, rgb_overlay_path=ov)
        u = env.unwrapped
        orig_xy = list(u._xy_configs); orig_q = list(u._quat_configs)
        if task == "Eggplant":
            settled, px = [], []
            for i in range(len(orig_xy) * len(orig_q)):
                xy0 = orig_xy[(i % (len(orig_xy) * len(orig_q))) // len(orig_q)]
                q0 = orig_q[i % len(orig_q)]
                saved = (u._xy_configs, u._quat_configs)
                u._xy_configs, u._quat_configs = [np.array(xy0)], [np.array(q0)]
                obs, _ = env.reset(options={"robot_init_options": {"init_xy": np.array(TASKS[task][4]),
                                                                   "init_rot_quat": np.array([0, 0, 0, 1])},
                                            "obj_init_options": {"episode_id": 0}})
                u._xy_configs, u._quat_configs = saved
                settled.append(np.array(u.episode_obj_xyzs_after_settle[0]))
                cam = [c for c in obs["image"] if "Segmentation" in obs["image"][c]]
                px.append(int((obs["image"][cam[-1]]["Segmentation"][..., 1] == u.episode_objs[0].id).sum()) if cam else -1)
            settled = np.array(settled); margin = np.array([0.045, 0.055])
            globals()["EGG_BOUNDS"] = (settled[:, :2].min(0) - margin, settled[:, :2].max(0) + margin,
                                       settled[:, 2].min() - 0.01, settled[:, 2].max() + 0.01,
                                       max(150, int(0.5 * np.median(px))), settled)
            print("  Eggplant: basin x %.3f..%.3f y %.3f..%.3f, pixel threshold %d"
                  % (EGG_BOUNDS[0][0], EGG_BOUNDS[1][0], EGG_BOUNDS[0][1], EGG_BOUNDS[1][1], EGG_BOUNDS[4]), flush=True)
        points = (np.array([x[0] for x in orig_xy]) if task == "Eggplant"
                  else np.unique(np.concatenate([np.array(x) for x in orig_xy]), axis=0))
        scenes, tried, rejected = [], 0, 0
        while len(scenes) < a.n and tried < a.n * 40:
            tried += 1
            base_xy = orig_xy[rng.randint(len(orig_xy))]
            base_q = orig_q[rng.randint(len(orig_q))]
            cand = propose(rng, task, base_xy, base_q, points)
            if cand is None:
                continue
            xy, q = cand
            ok, detail = check(env, task, xy, q)
            if not ok:
                rejected += 1
                continue
            scenes.append({"xy": np.round(xy, 4).tolist(), "quat": np.round(q, 6).tolist(), "check": detail})
        env.close()
        out[task] = scenes
        stats[task] = {"kept": len(scenes), "rejected": rejected, "tried": tried}
        print("%-10s kept %d scenes, rejected %d, tried %d" % (task, len(scenes), rejected, tried), flush=True)
    json.dump({"seed": a.seed, "scenes": out, "stats": stats}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
