#!/usr/bin/env python
"""Offline validator + UI->API converter for ComfyUI frontend workflows (frontend 1.52.x, ComfyUI v0.37.x).

Nothing here queues work: the only network call is a read-only GET /object_info (or a saved copy of it).

    python scripts/validate_comfy_workflow.py "path/to/workflow.json" [...] [--object-info object_info.json]
        [--server http://127.0.0.1:8188] [--input-dir D:/ComfyUI/input] [--write-api DIR] [--self-test TEMPLATES_DIR]

Checks, per workflow:
  * every node type exists in /object_info (frontend-only nodes such as Note / MarkdownNote are skipped),
  * every widget value is valid for its input spec (combo options, INT/FLOAT ranges and steps, BOOLEAN, STRING,
    dynamic combos and their nested children), and widgets_values / widgets_values_named agree,
  * every serialized input name exists in the spec (Autogrow inputs are expanded the way the backend does it:
    "<group>.<prefix><i>", i = 0..max-1),
  * the link table is consistent (both ends point at each other) and every link's output type is accepted by the
    input type (LiteGraph.isValidConnection rules),
  * after converting to the API format (bypassed / muted nodes dropped, links through bypassed nodes re-routed with
    the frontend's _getBypassSlotIndex rule, links to dropped nodes removed), every required input of every active node
    is present and every link still type-checks,
  * media files referenced by active LoadImage / LoadVideo / LoadAudio nodes exist in the input folder.

The widget order used to map widgets_values follows the frontend: inputs in input_order (required, then optional),
one widget per widget-type input, plus the frontend-only companions: "control_after_generate" after seed INTs,
"upload" after *_upload combos, an unnamed preview widget after audio_upload combos, and the selected option's
children after a COMFY_DYNAMICCOMBO_V3 ("<parent>.<child>").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

FRONTEND_ONLY = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "Group", "Label"}
WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
MODE_ALWAYS, MODE_NEVER, MODE_BYPASS = 0, 2, 4


# ------------------------------------------------------------------------------------------------- object_info
def load_object_info(path=None, server="http://127.0.0.1:8188"):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with urllib.request.urlopen(server.rstrip("/") + "/object_info", timeout=60) as r:  # read-only GET
        data = json.loads(r.read().decode("utf-8"))
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def _spec_opts(spec):
    return spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}


def _is_widget(spec):
    t = spec[0]
    if isinstance(t, list):
        return True
    if t == "COMFY_DYNAMICCOMBO_V3":
        return True
    if t in WIDGET_SCALARS:
        return not _spec_opts(spec).get("forceInput", False)
    return False


def _ordered_inputs(inputs_dict, order=None):
    """[(name, spec, optional)] in the frontend's order (input_order if given)."""
    out = []
    for kind in ("required", "optional"):
        d = inputs_dict.get(kind) or {}
        names = (order or {}).get(kind) or list(d.keys())
        for n in names:
            if n in d:
                out.append((n, d[n], kind == "optional"))
    return out


def node_spec(oi, node_type):
    info = oi[node_type]
    return _ordered_inputs(info.get("input", {}), info.get("input_order"))


# ------------------------------------------------------------------------------------------------- widgets
class W:
    """One frontend widget: name, spec (None for frontend-only companions), kind."""

    def __init__(self, name, spec, kind):
        self.name, self.spec, self.kind = name, spec, kind  # kind: value | control | upload | preview

    @property
    def to_api(self):
        return self.kind == "value"

    def __repr__(self):
        return f"W({self.name!r},{self.kind})"


def widget_list(oi, node_type, values_named=None, values=None):
    """Frontend widget list for a node; dynamic-combo children depend on the selected value, so values are consulted."""
    entries = node_spec(oi, node_type)
    out = []
    pos = [0]  # running index into positional values, for dynamic combo selection without named values

    def current_value(name, default):
        if values_named is not None and name in values_named:
            return values_named[name]
        if values is not None and pos[0] < len(values):
            return values[pos[0]]
        return default

    def add(name, spec, kind):
        out.append(W(name, spec, kind))
        pos[0] += 1

    def walk(items, prefix):
        for name, spec, _opt in items:
            if not _is_widget(spec):
                continue
            full = f"{prefix}{name}"
            t, o = spec[0], _spec_opts(spec)
            if t == "COMFY_DYNAMICCOMBO_V3":
                options = o.get("options", [])
                default = o.get("default", options[0]["key"] if options else None)
                sel = current_value(full, default)
                add(full, spec, "value")
                chosen = next((op for op in options if op["key"] == sel), None)
                if chosen is not None:
                    walk(_ordered_inputs(chosen.get("inputs", {})), full + ".")
                continue
            add(full, spec, "value")
            if t == "INT" and (o.get("control_after_generate") or name in ("seed", "noise_seed")):
                add("control_after_generate", None, "control")
            if o.get("image_upload") or o.get("video_upload") or o.get("animated_image_upload"):
                add("upload", None, "upload")
            elif o.get("audio_upload"):
                add("upload", None, "upload")
                add("", None, "preview")

    walk(entries, "")
    return out


# ------------------------------------------------------------------------------------------------- spec helpers
def expanded_input_specs(oi, node_type):
    """{input_name: (type, spec, optional)} for socket inputs, with Autogrow groups expanded backend-style."""
    res = {}
    for name, spec, optional in node_spec(oi, node_type):
        t = spec[0]
        o = _spec_opts(spec)
        if t == "COMFY_AUTOGROW_V3":
            tpl = o["template"]
            inner = tpl["input"]
            kind, d = next((k, v) for k, v in inner.items() if v)
            tname, tspec = next(iter(d.items()))
            if "names" in tpl:
                names = tpl["names"]
            else:
                names = [f"{tpl['prefix']}{i}" for i in range(tpl["max"])]
            for i, n in enumerate(names):
                req = (not optional) and kind == "required" and i < tpl.get("min", 0)
                res[f"{name}.{n}"] = (tspec[0], tspec, not req)
        else:
            res[name] = (t if isinstance(t, str) else "COMBO", spec, optional)
    return res


def type_ok(out_type, in_type):
    """LiteGraph.isValidConnection."""
    a, b = str(out_type or ""), str(in_type or "")
    if a in ("", "*", "0") or b in ("", "*", "0") or a == b:
        return True
    sa = {x.strip().lower() for x in a.split(",")}
    sb = {x.strip().lower() for x in b.split(",")}
    return bool(sa & sb)


def check_value(spec, value, input_dir=None):
    """Return an error string or None."""
    t, o = spec[0], _spec_opts(spec)
    if isinstance(t, list) or t == "COMBO":
        options = t if isinstance(t, list) else o.get("options", [])
        upload = o.get("image_upload") or o.get("video_upload") or o.get("audio_upload") or o.get("animated_image_upload")
        if upload:
            if value in options:
                return None
            if input_dir and isinstance(value, str) and os.path.isfile(os.path.join(input_dir, value)):
                return None
            return f"file {value!r} is not in the input folder"
        if value in options:
            return None
        return f"{value!r} not in options {options[:12]}{'...' if len(options) > 12 else ''}"
    if t == "COMFY_DYNAMICCOMBO_V3":
        keys = [op["key"] for op in o.get("options", [])]
        return None if value in keys else f"{value!r} not in dynamic combo keys {keys}"
    if t == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{value!r} is not an INT"
        lo, hi = o.get("min"), o.get("max")
        if lo is not None and value < lo:
            return f"{value} < min {lo}"
        if hi is not None and value > hi:
            return f"{value} > max {hi}"
        step = o.get("step")
        if step and step > 1:
            base = lo if lo is not None else 0
            if (value - base) % step:
                return f"{value} is off the step grid (min {base}, step {step})"
        return None
    if t == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{value!r} is not a FLOAT"
        lo, hi = o.get("min"), o.get("max")
        if lo is not None and value < lo:
            return f"{value} < min {lo}"
        if hi is not None and value > hi:
            return f"{value} > max {hi}"
        return None
    if t == "BOOLEAN":
        return None if isinstance(value, bool) else f"{value!r} is not a BOOLEAN"
    if t == "STRING":
        return None if isinstance(value, str) else f"{value!r} is not a STRING"
    return None


# ------------------------------------------------------------------------------------------------- validation
class Report:
    def __init__(self, name):
        self.name, self.errors, self.warnings, self.notes = name, [], [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def note(self, msg):
        self.notes.append(msg)

    def dump(self):
        lines = [f"== {self.name}: {'OK' if not self.errors else 'FAILED'} "
                 f"({len(self.errors)} errors, {len(self.warnings)} warnings)"]
        lines += [f"   ERROR   {m}" for m in self.errors]
        lines += [f"   warning {m}" for m in self.warnings]
        lines += [f"   info    {m}" for m in self.notes]
        return "\n".join(lines)


def _label(n):
    return f"#{n['id']} {n['type']}" + (f" '{n['title']}'" if n.get("title") else "")


def validate_workflow(wf, oi, input_dir=None, name="workflow"):
    rep = Report(name)
    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    if len(nodes) != len(wf.get("nodes", [])):
        rep.err("duplicate node ids")
    links = {}
    for l in wf.get("links", []):
        if l[0] in links:
            rep.err(f"duplicate link id {l[0]}")
        links[l[0]] = l
    if wf.get("last_node_id", 0) < max(nodes, default=0):
        rep.err("last_node_id is smaller than the largest node id")
    if wf.get("last_link_id", 0) < max(links, default=0):
        rep.err("last_link_id is smaller than the largest link id")

    # ---- node types, inputs, widgets
    for n in nodes.values():
        t = n["type"]
        if t in FRONTEND_ONLY:
            continue
        if t not in oi:
            rep.err(f"{_label(n)}: node type not in /object_info")
            continue
        specs = expanded_input_specs(oi, t)
        widgets = widget_list(oi, t, n.get("widgets_values_named"), n.get("widgets_values"))
        wnames = {w.name for w in widgets if w.name}
        for inp in n.get("inputs", []):
            nm = inp["name"]
            if inp.get("widget"):
                if inp["widget"].get("name") not in wnames:
                    rep.err(f"{_label(n)}: widget input {nm!r} is not a widget of this node")
                continue
            if nm not in specs:
                rep.err(f"{_label(n)}: input {nm!r} not in spec (have {sorted(specs)[:20]})")
                continue
            if not type_ok(inp.get("type"), specs[nm][0]):
                rep.err(f"{_label(n)}: input {nm!r} serialized type {inp.get('type')} != spec type {specs[nm][0]}")
        outs = oi[t]["output"]
        if len(n.get("outputs", [])) != len(outs):
            rep.err(f"{_label(n)}: {len(n.get('outputs', []))} outputs serialized, spec has {len(outs)}")
        for i, o in enumerate(n.get("outputs", [])):
            if i < len(outs) and not type_ok(o.get("type"), outs[i]):
                rep.err(f"{_label(n)}: output {i} type {o.get('type')} != spec {outs[i]}")

        vals = n.get("widgets_values")
        named = n.get("widgets_values_named")
        if widgets and vals is None and named is None:
            rep.err(f"{_label(n)}: has widgets but no widgets_values")
            continue
        if vals is not None and len(vals) != len(widgets):
            rep.err(f"{_label(n)}: {len(vals)} widgets_values for {len(widgets)} widgets {[w.name for w in widgets]}")
        if named is not None:
            expected = {w.name for w in widgets if w.kind in ("value", "control", "upload")}
            if set(named) != expected:
                rep.err(f"{_label(n)}: widgets_values_named keys {sorted(named)} != expected {sorted(expected)}")
        for i, w in enumerate(widgets):
            pv = vals[i] if vals is not None and i < len(vals) else None
            v = named[w.name] if named is not None and w.name in named else pv
            if named is not None and vals is not None and w.name in named and i < len(vals) and named[w.name] != vals[i]:
                rep.err(f"{_label(n)}: widget {w.name!r} positional {vals[i]!r} != named {named[w.name]!r}")
            if w.kind == "control":
                if v not in ("fixed", "increment", "decrement", "randomize"):
                    rep.err(f"{_label(n)}: control_after_generate {v!r} invalid")
                continue
            if w.kind != "value":
                continue
            # linked widget inputs take their value from the link
            if any(inp.get("widget", {}).get("name") == w.name and inp.get("link") is not None for inp in n.get("inputs", [])):
                continue
            e = check_value(w.spec, v, input_dir)
            if e:
                media = w.spec is not None and any(_spec_opts(w.spec).get(k) for k in ("image_upload", "video_upload", "audio_upload"))
                if media and n.get("mode") in (MODE_NEVER, MODE_BYPASS):
                    rep.warn(f"{_label(n)} (bypassed): {w.name}: {e}")
                else:
                    rep.err(f"{_label(n)}: {w.name}: {e}")

    # ---- link table
    for lid, (lid_, o_id, o_slot, t_id, t_slot, ltype) in links.items():
        on, tn = nodes.get(o_id), nodes.get(t_id)
        if on is None or tn is None:
            rep.err(f"link {lid}: missing node {o_id if on is None else t_id}")
            continue
        if o_slot >= len(on.get("outputs", [])):
            rep.err(f"link {lid}: origin slot {o_slot} out of range on {_label(on)}")
            continue
        if t_slot >= len(tn.get("inputs", [])):
            rep.err(f"link {lid}: target slot {t_slot} out of range on {_label(tn)}")
            continue
        out, inp = on["outputs"][o_slot], tn["inputs"][t_slot]
        if inp.get("link") != lid:
            rep.err(f"link {lid}: target input {inp['name']!r} of {_label(tn)} has link {inp.get('link')}")
        if lid not in (out.get("links") or []):
            rep.err(f"link {lid}: not listed in output {out['name']!r} of {_label(on)}")
        if ltype != out.get("type"):
            rep.err(f"link {lid}: link type {ltype} != output type {out.get('type')}")
        if not type_ok(out.get("type"), inp.get("type")):
            rep.err(f"link {lid}: {_label(on)}.{out['name']} ({out.get('type')}) -> {_label(tn)}.{inp['name']} ({inp.get('type')}) type mismatch")
    for n in nodes.values():
        for inp in n.get("inputs", []):
            if inp.get("link") is not None and inp["link"] not in links:
                rep.err(f"{_label(n)}: input {inp['name']!r} references missing link {inp['link']}")
        for o in n.get("outputs", []):
            for lid in o.get("links") or []:
                if lid not in links:
                    rep.err(f"{_label(n)}: output {o['name']!r} references missing link {lid}")

    # ---- API conversion and post-conversion checks
    api = None
    try:
        api = workflow_to_api(wf, oi)
    except Exception as e:  # pragma: no cover
        rep.err(f"API conversion failed: {e!r}")
        return rep, None
    for nid, node in api.items():
        t = node["class_type"]
        specs = expanded_input_specs(oi, t)
        widgets = {w.name: w for w in widget_list(oi, t, None, None)}
        # dynamic children present in API
        for nm, spec, optional in node_spec(oi, t):
            if not optional and nm not in node["inputs"]:
                rep.err(f"API #{nid} {t}: required input {nm!r} missing")
        for nm, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                src = api.get(v[0])
                if src is None:
                    rep.err(f"API #{nid} {t}: {nm} links to missing node {v[0]}")
                    continue
                st = oi[src["class_type"]]["output"][v[1]]
                it = specs[nm][0] if nm in specs else None
                if it is None:
                    rep.err(f"API #{nid} {t}: linked input {nm!r} not in spec")
                elif not type_ok(st, it):
                    rep.err(f"API #{nid} {t}: {nm} gets {st} from #{v[0]} {src['class_type']}, expects {it}")
            elif nm not in widgets and "." not in nm:
                rep.err(f"API #{nid} {t}: value input {nm!r} is not a widget")
        if t in ("LoadImage", "LoadVideo", "LoadAudio"):
            key = {"LoadImage": "image", "LoadVideo": "file", "LoadAudio": "audio"}[t]
            fn = node["inputs"].get(key)
            if input_dir and not os.path.isfile(os.path.join(input_dir, str(fn))):
                rep.err(f"API #{nid} {t}: active node needs {fn!r} in {input_dir}")
    outputs = [nid for nid, n in api.items() if oi[n["class_type"]].get("output_node")]
    if not outputs:
        rep.err("API graph has no active output node")
    rep.note(f"API: {len(api)} nodes, output nodes {outputs}")
    return rep, api


# ------------------------------------------------------------------------------------------------- UI -> API
def _live_inputs(oi, n):
    """Approximation of the live node's input slots (serialized sockets + one slot per widget input), for bypass matching."""
    slots = [(i["name"], i.get("type")) for i in n.get("inputs", [])]
    have = {s[0] for s in slots}
    if n["type"] in oi:
        for w in widget_list(oi, n["type"], n.get("widgets_values_named"), n.get("widgets_values")):
            if w.kind == "value" and w.name not in have and "." not in w.name:
                t = w.spec[0]
                slots.append((w.name, "COMBO" if isinstance(t, list) else t))
    return slots


def workflow_to_api(wf, oi):
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}

    def resolve(link_id, want_type, seen=None):
        """Follow a link to a real output, through bypassed nodes (frontend rule) and reroutes. None = dropped."""
        seen = seen or set()
        l = links.get(link_id)
        if l is None:
            return None
        _, o_id, o_slot, _, _, _ = l
        if (o_id, o_slot) in seen:
            raise RuntimeError(f"bypass cycle at node {o_id}")
        seen.add((o_id, o_slot))
        on = nodes[o_id]
        mode = on.get("mode", 0)
        if mode == MODE_NEVER:
            return None
        if on["type"] == "Reroute":
            inp = on.get("inputs", [{}])[0]
            return resolve(inp.get("link"), want_type, seen) if inp.get("link") is not None else None
        if mode == MODE_BYPASS:
            slots = _live_inputs(oi, on)
            out_type = on["outputs"][o_slot].get("type")
            idx = -1
            if want_type in ("*", ""):
                idx = o_slot if len(slots) > o_slot else 0
            elif o_slot < len(slots) and type_ok(slots[o_slot][1], out_type) and type_ok(slots[o_slot][1], want_type):
                idx = o_slot
            else:
                idx = next((i for i, s in enumerate(slots) if s[1] == want_type), -1)
                if idx == -1:
                    idx = next((i for i, s in enumerate(slots) if type_ok(s[1], out_type) and type_ok(s[1], want_type)), -1)
            if idx == -1 or idx >= len(on.get("inputs", [])):
                return None  # widget-only match or no match: the input is omitted
            nxt = on["inputs"][idx].get("link")
            return resolve(nxt, want_type, seen) if nxt is not None else None
        if on["type"] in FRONTEND_ONLY:
            return None
        return (str(o_id), o_slot)

    api = {}
    for nid, n in nodes.items():
        if n["type"] in FRONTEND_ONLY or n.get("mode", 0) in (MODE_NEVER, MODE_BYPASS):
            continue
        ins = {}
        widgets = widget_list(oi, n["type"], n.get("widgets_values_named"), n.get("widgets_values"))
        named = n.get("widgets_values_named")
        vals = n.get("widgets_values") or []
        for i, w in enumerate(widgets):
            if not w.to_api:
                continue
            if named is not None and w.name in named:
                v = named[w.name]
            elif i < len(vals):
                v = vals[i]
            else:
                continue
            ins[w.name] = {"__value__": v} if isinstance(v, list) else v
        for inp in n.get("inputs", []):
            if inp.get("link") is None:
                continue
            r = resolve(inp["link"], inp.get("type"))
            if r is None:  # dropped (muted / no bypass match): a linked widget keeps its own value, like the frontend
                continue
            ins[inp["name"]] = [r[0], r[1]]
        api[str(nid)] = {"inputs": ins, "class_type": n["type"], "_meta": {"title": n.get("title") or n["type"]}}
    # drop links to nodes that are not in the prompt (frontend does the same)
    for node in api.values():
        for k, v in list(node["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] not in api:
                del node["inputs"][k]
    return api


# ------------------------------------------------------------------------------------------------- self test
def self_test(oi, templates_dir):
    """Compare the derived widget names with widgets_values_named written by real frontend saves (templates)."""
    import glob

    checked, bad = 0, []
    for f in sorted(glob.glob(os.path.join(templates_dir, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        nodes = list(d.get("nodes", []))
        for sg in d.get("definitions", {}).get("subgraphs", []):
            nodes += sg.get("nodes", [])
        for n in nodes:
            t, named = n.get("type"), n.get("widgets_values_named")
            if named is None or t not in oi:
                continue
            ws = widget_list(oi, t, named, n.get("widgets_values"))
            exp = [w.name for w in ws if w.kind in ("value", "control", "upload")]
            checked += 1
            if set(exp) != set(named):
                bad.append((os.path.basename(f), t, sorted(named), sorted(exp)))
    return checked, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflows", nargs="*")
    ap.add_argument("--object-info", default=None, help="saved /object_info JSON (fetched from --server and saved here if missing)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--input-dir", default=None, help="ComfyUI input folder, to check media files")
    ap.add_argument("--write-api", default=None, metavar="DIR",
                    help="write <name>.api.json into DIR (never next to a workflow: in ComfyUI's workflows folder it would show up as a broken workflow)")
    ap.add_argument("--self-test", default=None, metavar="TEMPLATES_DIR")
    a = ap.parse_args()
    oi = load_object_info(a.object_info, a.server)
    ok = True
    if a.self_test:
        checked, bad = self_test(oi, a.self_test)
        print(f"== self-test: widget names derived for {checked} template nodes saved by the real frontend; {len(bad)} mismatches")
        for b in bad[:30]:
            print("   mismatch", b)
        # mismatches on nodes whose schema changed after the template was saved are expected; report, don't fail
    for path in a.workflows:
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        rep, api = validate_workflow(wf, oi, a.input_dir, os.path.basename(path))
        print(rep.dump())
        ok &= not rep.errors
        if a.write_api and api is not None:
            os.makedirs(a.write_api, exist_ok=True)
            out = os.path.join(a.write_api, os.path.splitext(os.path.basename(path))[0] + ".api.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(api, f, indent=1, ensure_ascii=False)
            print(f"   wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
