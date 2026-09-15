"""
Experiment 7: Deployment cost

Deployment cost is important because, a model that wins a benchmark and cannot
run inside the latency, memory and power budget of a flight processor is not a
candidate. This experiment measures the quantities that decide that:

  parameters      how much memory the weights occupy
  model size      the same thing on disk, before and after quantisation
  MACs            multiply-accumulate operations per inference, the hardware-
                  independent measure of arithmetic work
  latency         wall-clock time per frame, measured properly
  peak memory     activation memory during a forward pass
  MC cost         the multiplier that Monte Carlo dropout imposes

The single-thread CPU number is the ideal scenario. Flight processors are
radiation-hardened and slow: an ESA GR740 runs four LEON4 cores at 250 MHz with no
GPU and no wide vector units. Google colab CPU is used here, so a model that is already slow single-threaded here is not considered.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn



# Size

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def model_size_mb(model, path="/tmp/_size_probe.pt"):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return mb



# Arithmetic work

def count_macs(model, input_shape, device="cpu"):
    """Multiply-accumulate operations for one forward pass.

    Counted with forward hooks on the layer types that dominate: convolutions,
    linear layers and the attention matmuls inside a transformer block. This is
    an approximation, and it is the same approximation applied to both models,
    so the comparison between them is fair even where the absolute number is
    not exact.

    One MAC is one multiply plus one add. Papers that quote FLOPs usually mean
    2 x MACs; both are reported so the number can be compared either way.
    """
    macs = [0]
    hooks = []

    def conv_hook(m, inp, out):
        # output elements x (kernel area x input channels / groups)
        out_elems = out.numel()
        k = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += out_elems * k * (m.in_channels // m.groups)

    def linear_hook(m, inp, out):
        macs[0] += out.numel() * m.in_features

    def attn_hook(m, inp, out):
        # q@k^T and attn@v, both (tokens x tokens x dim)
        x = inp[0]
        if x.dim() == 3:
            b, n, d = x.shape
            macs[0] += 2 * b * n * n * d

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif m.__class__.__name__.endswith("SelfAttention"):
            hooks.append(m.register_forward_hook(attn_hook))

    model.eval().to(device)
    with torch.no_grad():
        model(torch.zeros(*input_shape, device=device))
    for h in hooks:
        h.remove()
    return macs[0]



# Latency

def measure_latency(model, input_shape, device="cpu", n_warmup=10, n_runs=50,
                    threads=None):
    """Per-frame latency, batch size 1.

    
      - the first calls include lazy initialisation and cache warming, so a
        warm-up loop runs first and is discarded
      - CUDA calls are asynchronous, so the GPU must be synchronised before
        reading the clock or the measurement is meaningless
      - a single run is noisy, so the median over many runs is reported along
        with the spread

    Batch size is 1 because a navigation system processes one frame at a time.
    Throughput at large batch is irrelevant to it.
    """
    old_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)

    model.eval().to(device)
    x = torch.zeros(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    torch.set_num_threads(old_threads)
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "mean_ms": float(t.mean()),
            "p95_ms": float(np.percentile(t, 95)), "std_ms": float(t.std()),
            "fps": float(1000.0 / np.median(t))}


def measure_peak_memory(model, input_shape, device="cpu"):
    """Peak activation memory for one forward pass, in MB. CUDA only."""
    if device != "cuda":
        return None
    model.eval().to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        model(torch.zeros(*input_shape, device=device))
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / 1e6



# Quantisation

def quantize_dynamic_int8(model):
    """Dynamic INT8 quantisation of the linear layers, on CPU.

    Weights are stored as 8-bit integers and activations are quantised on the
    fly. This is the cheapest form of quantisation: no calibration data and no
    retraining. It only touches Linear layers, so it helps a transformer far
    more than a convolutional network, whose work sits in Conv2d.

    Flight implementations go further, with mixed-precision quantisation mapped
    onto FPGA fabric. This is a lower bound on what quantisation can achieve,
    not an upper one.
    """
    m = torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    return m


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def profile_model(model, name, input_shape=(1, 1, 224, 224), device="cpu",
                  n_runs=50, log=print):
    """Full deployment profile of one model."""
    total, train = count_parameters(model)
    size = model_size_mb(model)
    try:
        macs = count_macs(model, input_shape, device="cpu")
    except Exception as e:
        log(f"    MAC counting failed: {type(e).__name__}")
        macs = float("nan")

    row = {"name": name, "params_M": total / 1e6, "trainable_M": train / 1e6,
           "size_MB": size, "MACs_G": macs / 1e9, "GFLOPs": 2 * macs / 1e9}

    lat_cpu1 = measure_latency(model, input_shape, "cpu", n_runs=n_runs, threads=1)
    row["cpu1_ms"] = lat_cpu1["median_ms"]
    row["cpu1_p95_ms"] = lat_cpu1["p95_ms"]

    lat_cpu = measure_latency(model, input_shape, "cpu", n_runs=n_runs)
    row["cpu_ms"] = lat_cpu["median_ms"]

    if device == "cuda" and torch.cuda.is_available():
        lat_gpu = measure_latency(model, input_shape, "cuda", n_runs=n_runs)
        row["gpu_ms"] = lat_gpu["median_ms"]
        row["gpu_fps"] = lat_gpu["fps"]
        row["peak_mem_MB"] = measure_peak_memory(model, input_shape, "cuda")
    else:
        row["gpu_ms"] = float("nan"); row["gpu_fps"] = float("nan")
        row["peak_mem_MB"] = float("nan")

    return row


def print_profile_table(rows, log=print):
    hdr = (f"{'model':<22}{'params M':>10}{'size MB':>10}{'GMACs':>9}"
           f"{'GPU ms':>9}{'CPU ms':>9}{'CPU-1t ms':>11}")
    log(hdr); log("-" * len(hdr))
    for r in rows:
        log(f"{r['name']:<22}{r['params_M']:>10.2f}{r['size_MB']:>10.1f}"
            f"{r['MACs_G']:>9.2f}{r['gpu_ms']:>9.1f}{r['cpu_ms']:>9.1f}"
            f"{r['cpu1_ms']:>11.1f}")


def mc_dropout_cost(base_latency_ms, n_mc):
    """Monte Carlo dropout runs the whole network n_mc times.

    This is the part of an uncertainty estimate that is easy to forget when
    reporting accuracy. If a model needs 25 stochastic passes to produce an
    epistemic estimate, its effective latency is 25 times the single-pass
    number, and the deployment question has to be asked again at that figure.
    """
    return {"single_pass_ms": base_latency_ms,
            "n_mc": n_mc,
            "total_ms": base_latency_ms * n_mc,
            "effective_fps": 1000.0 / (base_latency_ms * n_mc)}


def budget_check(latency_ms, camera_hz=1.0, budget_fraction=0.5, log=print):
    """Does the model fit the time available between frames?

    A navigation function does more than run a network: it detects the target,
    solves geometry, runs the filter and hands a state to guidance and control.
    Taking half the inter-frame interval as the perception budget is a working
    assumption, not a requirement from any specification, and it is stated that
    way.
    """
    interval_ms = 1000.0 / camera_hz
    budget_ms = interval_ms * budget_fraction
    fits = latency_ms < budget_ms
    log(f"  camera at {camera_hz:g} Hz -> {interval_ms:.0f} ms between frames")
    log(f"  assumed perception budget ({budget_fraction:.0%}): {budget_ms:.0f} ms")
    log(f"  measured latency: {latency_ms:.1f} ms -> "
        f"{'fits' if fits else 'DOES NOT FIT'} "
        f"({latency_ms/budget_ms:.2f}x the budget)")
    return fits


# ---------------------------------------------------------------------------
# Flight processor context
# ---------------------------------------------------------------------------
FLIGHT_PROCESSORS = [
    # name, description, rough integer throughput, notes
    ("GR740 (LEON4FT)", "ESA next-generation quad-core SPARC, 250 MHz",
     "~1700 DMIPS", "radiation hardened, no GPU, no wide SIMD"),
    ("GR712RC (LEON3FT)", "dual-core SPARC, 100 MHz",
     "~200 DMIPS", "radiation hardened, widely flown"),
    ("RAD750", "PowerPC, 133-200 MHz",
     "~400 DMIPS", "radiation hardened, long flight heritage"),
    ("Zynq UltraScale+ MPSoC", "COTS FPGA + ARM, used with mitigation",
     "FPGA fabric", "the usual route for CNN inference in recent work"),
    ("Myriad 2 VPU", "COTS vision processor, flown on Phi-sat-1",
     "~100 GFLOPS", "not radiation hardened; first European onboard AI demo"),
]


def print_flight_context(log=print):
    log("Representative onboard processors:")
    log(f"  {'processor':<26}{'description':<46}{'throughput':<16}")
    for n, d, t, notes in FLIGHT_PROCESSORS:
        log(f"  {n:<26}{d:<46}{t:<16}")
    log("")
    log("  A modern laptop CPU is roughly two to three orders of magnitude")
    log("  faster than a GR740 for this kind of work. The single-thread CPU")
    log("  latency measured here is therefore an optimistic proxy: a model")
    log("  that is already slow in that column is not a flight candidate")
    log("  without dedicated acceleration.")
