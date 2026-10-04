import io, os, shutil

HOME = os.environ.get("CORRECTOR_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = HOME + "/dboft/dexbotic-benchmark/evaluation/utils/simpler_maniskill2.py"
s = io.open(p).read()
if "WPFS_TRAIN_SCENES" in s:
    print("client: already patched (WPFS_TRAIN_SCENES)"); raise SystemExit
shutil.copy(p, p + ".orig_scenes")
anchor = """    env_reset_options = {
        "robot_init_options": {"""
assert s.count(anchor) == 1, "patch site not found"
patch = '''    _ns = os.environ.get("DBOFT_SCENES")
    if _ns:
        import json as _json
        import numpy as _np
        _task = ("StackCube" if "StackGreenCube" in env_name else
                 "Carrot" if "Carrot" in env_name else
                 "Spoon" if "Spoon" in env_name else "Eggplant")
        _scenes = _json.load(open(_ns))["scenes"][_task]
        _u = env.unwrapped
        _orig_reset = _u.reset

        def _reset_train_scene(seed=None, options=None, _u=_u, _scenes=_scenes, _orig=_orig_reset):
            options = dict(options or {})
            _oi = dict(options.get("obj_init_options", {}))
            _eid = int(_oi.get("episode_id", 0))
            _c = _scenes[_eid % len(_scenes)]
            _u._xy_configs = [_np.array(_c["xy"], dtype=_np.float32)]
            _u._quat_configs = [_np.array(_c["quat"], dtype=_np.float32)]
            _oi["episode_id"] = 0
            options["obj_init_options"] = _oi
            _res = _orig(seed=seed, options=options)
            _res[1].update({"episode_id": _eid, "train_scene": True})
            return _res

        _u.reset = _reset_train_scene
        logger.info("WPFS_TRAIN_SCENES: %d training scenes from %s", len(_scenes), _ns)

'''
s = s.replace(anchor, patch + anchor)
if "\nimport os" not in s and not s.startswith("import os"):
    s = "import os\n" + s
compile(s, p, "exec")
io.open(p, "w").write(s)
print("client: patched WPFS_TRAIN_SCENES (backup:", p + ".orig_scenes)")
