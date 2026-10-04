import io, os, shutil

HOME = os.environ.get("CORRECTOR_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = HOME + "/dboft"
BENCH = D + "/dexbotic-benchmark"

link = BENCH + "/simpler"
if not os.path.exists(link + "/ManiSkill2_real2sim"):
    if os.path.isdir(link) and not os.listdir(link):
        os.rmdir(link)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(D + "/SimplerEnv", link)
print("simpler ->", os.path.realpath(link))

CFG = """defaults:
  - base_config

replan_step: 5
base_url: http://127.0.0.1:7891
use_delta: true
action_ensemble_horizon: 7
adaptive_ensemble_alpha: 0.1
action_ensemble: false
{extra}
output_dir: {out}
"""
cdir = BENCH + "/evaluation/configs/simpler"
for name, extra in (("dboft_local.yaml", ""), ("dboft_calib.yaml", "obj_episode_range: [0, 4]\n")):
    io.open(os.path.join(cdir, name), "w").write(CFG.format(extra=extra, out=D + "/results/dboft"))
    print("wrote", os.path.join(cdir, name))

p = BENCH + "/evaluation/policies/simpler_vla_agent.py"
s = io.open(p).read()
if "DBOFT_VARSEG" in s:
    print("client: already patched (DBOFT_VARSEG)")
else:
    old = "        for i in range(self.replan_step):"
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old,
                  "        n_act = len(raw_actions) if os.environ.get('DBOFT_VARSEG') == '1' else self.replan_step\n"
                  "        for i in range(n_act):")
    if "\nimport os" not in s and not s.startswith("import os"):
        s = "import os\n" + s
    compile(s, p, "exec")
    shutil.copy2(p, p + ".orig_varseg")
    io.open(p, "w").write(s)
    print("client: patched DBOFT_VARSEG (backup .orig_varseg)")
