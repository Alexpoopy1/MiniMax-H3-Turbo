"""comfy_h3_vae.py (page-locked RAM arena for the H3 VAEs), tested on the CPU against small fake ComfyUI modules.

* only the layers ComfyUI streams at decode time go into the arena (decoder side, cast layers above 16 KiB), with the same
  values; ComfyUI's pin_memory skips them; encoder, tiny layers, non-cast parameters and buffers are left alone;
* a partial lock takes whole layers, as many as fit above the free-RAM floor; a tensor shared by two layers is copied once;
* ComfyUI making room to pin another model does not take the arena; RAM pressure after a node, or critically low RAM, does,
  and then every layer gets its original (file-backed) tensor back and is locked again at a later decode;
* an arena whose tensors ComfyUI replaced is given back for good; without dynamic VRAM nothing is registered or reserved;
* a failure in the middle of the copy leaves every layer untouched; checkpoint pages are marked for unmapping and bounced.
"""
import importlib.util
import os
import sys
import types

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 1 << 30


class _CudaRT:
    def __init__(self):
        self.registered, self.unregistered = [], []

    def cudaHostRegister(self, ptr, size, flags):
        self.registered.append((ptr, size))
        return 0

    def cudaHostUnregister(self, ptr):
        self.unregistered.append(ptr)
        return 0


class CastLinear(torch.nn.Linear):  # stands in for a comfy.ops layer
    comfy_cast_weights = False


class MiniMaxH3VideoVAE(torch.nn.Module):
    def __init__(self, share=False):
        super().__init__()
        self.encoder = CastLinear(128, 128)  # 64.5 KiB, encoder side: never locked
        self.post_quant_conv = CastLinear(128, 128)
        self.decoder = torch.nn.Sequential(
            CastLinear(128, 128),
            CastLinear(128, 128),
            CastLinear(8, 8),  # tiny: ComfyUI force-loads it to the GPU
            torch.nn.Linear(128, 128),  # not a cast layer: force-loaded too
        )
        if share:
            self.decoder[1].weight = self.decoder[0].weight
        self.register_buffer("table", torch.arange(32, dtype=torch.float32))

    def big(self):
        return [self.post_quant_conv, self.decoder[0], self.decoder[1]]

    def others(self):
        return [self.encoder, self.decoder[2], self.decoder[3]]


def _nbytes(t):
    return -(-(t.numel() * t.element_size()) // 256) * 256


def _layer_bytes(m):
    return _nbytes(m.weight) + _nbytes(m.bias)


class _FakeComfy:
    """Just what comfy_h3_vae touches, with free RAM and ComfyUI's own pin eviction under the test's control."""

    def __init__(self, available, aimdo=True):
        self.available = available
        self.orig_pin_calls, self.orig_free_calls, self.orig_free_returns, self.dirty = [], [], 0, []
        comfy = types.ModuleType("comfy")
        mem = types.ModuleType("comfy.memory_management")
        mem.aimdo_enabled = aimdo
        mem.RAM_CACHE_HEADROOM = int(3.19 * GIB)
        sysmem = types.ModuleType("comfy.system_memory")
        sysmem.virtual_memory_available = lambda: self.available
        mm = types.ModuleType("comfy.model_management")
        mm.discard_cuda_async_error = lambda: None
        mm.mark_mmap_dirty = lambda storage: self.dirty.append(storage)

        def free_pins(size, evict_active=False, loaded=False):
            self.orig_free_calls.append(size)
            return self.orig_free_returns

        def ensure_pin_budget(size, evict_active=False, loaded=False):  # like ComfyUI: frees pins through the module global
            return mm.free_pins(size) >= size

        mm.free_pins, mm.ensure_pin_budget = free_pins, ensure_pin_budget
        pm = types.ModuleType("comfy.pinned_memory")

        def pin_memory(module, subset="weights", size=None):
            self.orig_pin_calls.append(module)
            return True

        pm.pin_memory = pin_memory
        sd = types.ModuleType("comfy.sd")

        class VAE:
            def __init__(self, model):
                self.first_stage_model = model

            def decode(self, x):
                return x

            def decode_tiled(self, x):
                return x

        sd.VAE = VAE
        ldm = types.ModuleType("comfy.ldm")
        minimax = types.ModuleType("comfy.ldm.minimax")
        vae_mod = types.ModuleType("comfy.ldm.minimax.vae")
        vae_mod.MiniMaxH3VideoVAE = MiniMaxH3VideoVAE
        self.modules = {
            "comfy": comfy, "comfy.memory_management": mem, "comfy.system_memory": sysmem, "comfy.model_management": mm,
            "comfy.pinned_memory": pm, "comfy.sd": sd, "comfy.ldm": ldm, "comfy.ldm.minimax": minimax,
            "comfy.ldm.minimax.vae": vae_mod,  # no audio_vae module: install() must still lock the video VAE
        }
        for name, m in self.modules.items():
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(self.modules[parent], child, m)
        self.mm, self.pm, self.sd = mm, pm, sd


@pytest.fixture
def env(monkeypatch):
    def make(available, aimdo=True):
        fake = _FakeComfy(available, aimdo)
        for name, m in fake.modules.items():
            monkeypatch.setitem(sys.modules, name, m)
        rt = _CudaRT()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "cudart", lambda: rt)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        spec = importlib.util.spec_from_file_location("comfy_h3_vae_under_test", os.path.join(_ROOT, "comfy_h3_vae.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # a fresh copy per test: its registries and the hooks start empty
        mod._GC_EVERY_S = 1e9  # no gc.collect() in tests
        mod.install()
        fake.rt, fake.lock = rt, mod
        return fake

    return make


def _snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _range(vae):
    a = vae._h3turbo_arena
    return (a.data_ptr(), a.data_ptr() + a.numel())


def _inside(rng, t):
    return rng[0] <= t.data_ptr() < rng[1]


def _locked_vae(f, share=False):
    model = MiniMaxH3VideoVAE(share=share)
    vae = f.sd.VAE(model)
    vae.decode(torch.zeros(1))
    return model, vae


def test_full_lock_decoder_layers_only(env):
    f = env(available=8 * GIB)
    model = MiniMaxH3VideoVAE()
    before = _snapshot(model)
    vae = f.sd.VAE(model)
    want = sum(_layer_bytes(m) for m in model.big())
    assert f.lock.pending_reserve_bytes() == want
    vae.decode(torch.zeros(1))
    assert vae._h3turbo_arena.numel() == want and f.lock.pending_reserve_bytes() == 0
    rng = _range(vae)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    for m in model.big():
        assert _inside(rng, m.weight) and _inside(rng, m.bias) and m._h3turbo_locked
        assert isinstance(m.weight, torch.nn.Parameter)
    for m in model.others():
        assert not _inside(rng, m.weight) and not getattr(m, "_h3turbo_locked", False)
    assert not _inside(rng, model.table)
    assert f.rt.registered == [(rng[0], want)]
    assert f.pm.pin_memory(model.decoder[0]) is None and f.orig_pin_calls == []  # locked: ComfyUI stages nothing
    assert f.pm.pin_memory(model.encoder) is True and f.orig_pin_calls == [model.encoder]


def test_partial_lock_whole_layers(env):
    f = env(available=0)
    model = MiniMaxH3VideoVAE()
    f.available = 2 * GIB + 2 * _layer_bytes(model.decoder[0]) + 100  # room for two of the three big layers
    f.lock._MIN_LOCK = 0
    vae = f.sd.VAE(model)
    vae.decode(torch.zeros(1))
    rng = _range(vae)
    assert _inside(rng, model.post_quant_conv.weight) and _inside(rng, model.decoder[0].weight)
    assert not _inside(rng, model.decoder[1].weight) and not getattr(model.decoder[1], "_h3turbo_locked", False)
    assert f.pm.pin_memory(model.decoder[1]) is True  # the rest keeps ComfyUI's default path
    assert f.lock.pending_reserve_bytes() == 0


def test_skip_below_floor_then_retry(env):
    f = env(available=2 * GIB)
    model, vae = _locked_vae(f)
    assert getattr(vae, "_h3turbo_arena", None) is None and f.lock.pending_reserve_bytes() > 0
    f.available = 8 * GIB
    vae.decode(torch.zeros(1))
    assert vae._h3turbo_arena is not None


def test_shared_tensor_copied_once(env):
    f = env(available=8 * GIB)
    model = MiniMaxH3VideoVAE(share=True)
    vae = f.sd.VAE(model)
    vae.decode_tiled(torch.zeros(1))
    assert model.decoder[0].weight is model.decoder[1].weight and _inside(_range(vae), model.decoder[0].weight)
    used = _layer_bytes(model.post_quant_conv) + _layer_bytes(model.decoder[0]) + _nbytes(model.decoder[1].bias)
    assert f.lock.pending_reserve_bytes() == 0 and used <= vae._h3turbo_arena.numel()


def test_release_rules_and_restore(env):
    f = env(available=8 * GIB)
    model = MiniMaxH3VideoVAE()
    before = _snapshot(model)
    orig_ptrs = {k: v.data_ptr() for k, v in model.state_dict().items()}
    vae = f.sd.VAE(model)
    vae.decode(torch.zeros(1))
    n = vae._h3turbo_arena.numel()
    rng = _range(vae)
    # ComfyUI making room to pin the DiT (asks for its whole size) while RAM is fine: its own pins go, the arena stays
    f.available = 3 * GIB
    assert f.mm.ensure_pin_budget(12 * GIB) is False and vae._h3turbo_arena is not None
    # RAM pressure after a node (execution.py calls free_pins directly): given back, every layer gets its own tensor again
    assert f.mm.free_pins(GIB) == n
    assert getattr(vae, "_h3turbo_arena", None) is None and f.lock.pending_reserve_bytes() == n
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]) and not _inside(rng, v) and v.data_ptr() == orig_ptrs[k]
    assert not any(getattr(m, "_h3turbo_locked", False) for m in model.modules())
    # locked again at a later decode; then critically low RAM takes it even while making room
    f.available = 8 * GIB
    vae.decode(torch.zeros(1))
    assert vae._h3turbo_arena is not None
    f.available = GIB // 2
    f.mm.ensure_pin_budget(GIB)
    assert getattr(vae, "_h3turbo_arena", None) is None


def test_replaced_arena_released_for_good(env):
    f = env(available=8 * GIB)
    model, vae = _locked_vae(f)
    rng = _range(vae)
    with torch.no_grad():  # ComfyUI swapping the big weights for its own copies
        for m in model.big():
            m.weight.data = m.weight.data.clone()
    vae.decode(torch.zeros(1))
    assert getattr(vae, "_h3turbo_arena", None) is None and f.lock.pending_reserve_bytes() == 0
    assert not any(_inside(rng, m.bias) for m in model.big())  # what was still in the arena got its original back
    vae.decode(torch.zeros(1))
    assert getattr(vae, "_h3turbo_arena", None) is None  # not retried


def test_small_moves_do_not_count_as_replaced(env):
    f = env(available=8 * GIB)
    model, vae = _locked_vae(f)
    with torch.no_grad():
        model.decoder[0].bias.data = model.decoder[0].bias.data.clone()
    vae.decode(torch.zeros(1))
    assert vae._h3turbo_arena is not None and _inside(_range(vae), model.decoder[0].weight)


def test_nothing_reserved_without_dynamic_vram(env):
    f = env(available=8 * GIB, aimdo=False)
    model, vae = _locked_vae(f)
    assert f.lock.pending_reserve_bytes() == 0 and getattr(vae, "_h3turbo_arena", None) is None


def test_failure_mid_copy_touches_nothing(env):
    f = env(available=8 * GIB)
    model = MiniMaxH3VideoVAE()
    ptrs = {k: v.data_ptr() for k, v in model.state_dict().items()}
    model.decoder[1].weight.data.untyped_storage()._comfy_tensor_mmap_refs = (object(), None)

    def boom(storage):
        raise RuntimeError("mapping gone")

    f.mm.mark_mmap_dirty = boom
    vae = f.sd.VAE(model)
    vae.decode(torch.zeros(1))
    assert getattr(vae, "_h3turbo_arena", None) is None and f.lock.pending_reserve_bytes() == 0
    assert {k: v.data_ptr() for k, v in model.state_dict().items()} == ptrs
    assert not any(getattr(m, "_h3turbo_locked", False) for m in model.modules())


def test_mapped_pages_marked_and_bounced(env):
    f = env(available=8 * GIB)
    model = MiniMaxH3VideoVAE()

    class Map:
        bounces = 0

        def bounce(self):
            Map.bounces += 1

    for m in model.big():
        m.weight.data.untyped_storage()._comfy_tensor_mmap_refs = (Map(), None)
    f.lock._BOUNCE_EVERY = 1
    vae = f.sd.VAE(model)
    vae.decode(torch.zeros(1))
    assert vae._h3turbo_arena is not None and len(f.dirty) == 3 and Map.bounces == 3
