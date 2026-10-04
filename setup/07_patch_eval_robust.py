import ast
import io
import os
import shutil

HOME = os.environ.get("CORRECTOR_HOME",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = HOME + "/dboft/dexbotic-benchmark"


def patch_step_errors():
    p = BENCH + "/evaluation/utils/simpler_maniskill2.py"
    s = io.open(p).read()
    if "WPFS_ROBUST_STEP" in s:
        print("  simpler_maniskill2.py: already patched"); return
    shutil.copy(p, p + ".orig_robust_step")
    a = "    predicted_terminated, done, truncated = False, False, False"
    assert s.count(a) == 1, "episode-loop initialisation not found"
    s = s.replace(a, a + "\n    info = {}\n    _WPFS_ROBUST_STEP = True")
    b = """        except Exception as e:
            logger.error(f"Step {timestep} execution failed: {e}")
            break"""
    assert s.count(b) == 1, "episode-loop except branch not found"
    s = s.replace(b, """        except Exception as e:
            import traceback as _tb
            logger.error("Step %d execution failed: %s\\n%s", timestep, e, _tb.format_exc())
            success = "failure"
            break""")
    ast.parse(s)
    io.open(p, "w").write(s)
    print("  simpler_maniskill2.py: patched (backup .orig_robust_step)")


def patch_action_shape():
    p = BENCH + "/evaluation/policies/simpler_vla_agent.py"
    s = io.open(p).read()
    if "WPFS_ACTION_SHAPE" in s:
        print("  simpler_vla_agent.py: already patched"); return
    shutil.copy(p, p + ".orig_action_shape")
    a = "        raw_actions = np.array(raw_actions)"
    assert s.count(a) == 1, "raw_actions conversion not found"
    s = s.replace(a, a + """
        if raw_actions.ndim == 1 and raw_actions.size >= 7:
            raw_actions = raw_actions.reshape(1, -1)
        if raw_actions.ndim != 2 or raw_actions.shape[0] == 0 or raw_actions.shape[1] < 7:
            import logging as _lg
            _lg.getLogger(__name__).error(
                "WPFS_ACTION_SHAPE: server returned shape %s - skipping this call",
                getattr(raw_actions, "shape", None))
            return""")
    ast.parse(s)
    io.open(p, "w").write(s)
    print("  simpler_vla_agent.py: patched (backup .orig_action_shape)")


if __name__ == "__main__":
    print("patching", BENCH)
    patch_step_errors()
    patch_action_shape()
    print("done")
