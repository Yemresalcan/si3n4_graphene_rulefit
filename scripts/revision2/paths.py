"""Locate the dataset, the analysis output and the figure output directory.

The same scripts are run from two layouts: the authors' working directory and
the public repository. Rather than hard-coding either, each location is resolved
by searching upwards from this file for a known landmark, so a reader who clones
the repository can execute every script without editing a path. Each location
can also be overridden with an environment variable.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [HERE] + [os.path.abspath(os.path.join(HERE, *[".."] * n)) for n in (1, 2, 3)]


def _resolve(candidates, description, env_var):
    override = os.environ.get(env_var)
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError("%s=%s does not exist" % (env_var, override))
        return override
    for root in _ROOTS:
        for rel in candidates:
            path = os.path.normpath(os.path.join(root, rel))
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        "could not locate %s; searched %s under %s. Set %s to override."
        % (description, candidates, _ROOTS, env_var)
    )


#: the 82-sample Vickers hardness dataset
DATA = _resolve(
    ["data.xlsx", os.path.join("data", "data.xlsx")],
    "data.xlsx", "SI3N4_DATA",
)

#: directory holding the released analysis output (paper_*.csv, rulefit_top_rules.csv)
RESULTS = _resolve(
    ["results", os.path.join("results", "tables")],
    "the results directory", "SI3N4_RESULTS",
)
if not os.path.exists(os.path.join(RESULTS, "rulefit_top_rules.csv")):
    RESULTS = _resolve(
        [os.path.join("results", "tables")], "the results tables directory", "SI3N4_RESULTS"
    )


def _default_out():
    for root in _ROOTS:
        candidate = os.path.join(root, "results", "revision2")
        if os.path.isdir(candidate):
            return candidate
    return os.path.abspath(os.path.join(HERE, ".."))


#: where regenerated figures and tables are written
OUT = os.environ.get("SI3N4_OUT") or _default_out()
FIG_OUT = os.path.join(OUT, "figures") if os.path.isdir(os.path.join(OUT, "figures")) else OUT
TABLE_OUT = os.path.join(OUT, "tables") if os.path.isdir(os.path.join(OUT, "tables")) else OUT


def results_file(name):
    return os.path.join(RESULTS, name)


if __name__ == "__main__":
    for label, value in (("DATA", DATA), ("RESULTS", RESULTS), ("OUT", OUT),
                         ("FIG_OUT", FIG_OUT), ("TABLE_OUT", TABLE_OUT)):
        print("%-10s %s" % (label, value))
