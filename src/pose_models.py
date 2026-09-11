"""
Pose regression models with predictive uncertainty.

Two backbones are compared on the same head and the same training recipe:

  CNN   a ResNet-18, the standard convolutional baseline
  ViT   DINOv2 ViT-S/14, a self-supervised vision transformer

The head predicts three things:
  t_mu      relative translation, in metres
  t_logvar  a per-axis log-variance for that translation (aleatoric)
  q         relative attitude as a scalar-first quaternion

The translation variance is trained with a Gaussian negative log-likelihood
(Kendall and Gal, NeurIPS 2017), so the network can widen its own error bars on
a hard image but pays for doing so. Dropout sits in the head and is left active
at inference to give the Monte Carlo dropout estimate of epistemic uncertainty
(Gal and Ghahramani, ICML 2016).

Predicting a covariance for translation, rather than a bare number, is the point:
it is what a relative navigation filter needs as its measurement noise R.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# Backbones

class SmallCNN(nn.Module):
    """Fallback convolutional trunk if torchvision is unavailable."""

    def __init__(self, width=32, in_ch=1):
        super().__init__()
        w = width

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(in_ch, w), blk(w, 2 * w), blk(2 * w, 4 * w),
                                 blk(4 * w, 8 * w), nn.AdaptiveAvgPool2d(1))
        self.out_dim = 8 * w

    def forward(self, x):
        return self.net(x).flatten(1)


def build_cnn_backbone(pretrained=True, in_ch=1):
    try:
        import torchvision
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        if in_ch == 1:
            w = m.conv1.weight.data.sum(1, keepdim=True)
            m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            m.conv1.weight.data = w
        dim = m.fc.in_features
        m.fc = nn.Identity()

        class Wrap(nn.Module):
            def __init__(self, m, d):
                super().__init__(); self.m = m; self.out_dim = d

            def forward(self, x):
                return self.m(x)
        return Wrap(m, dim), "resnet18"
    except Exception as e:
        print("torchvision unavailable, using SmallCNN:", type(e).__name__)
        b = SmallCNN(in_ch=in_ch)
        return b, "smallcnn"


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT trunk. Expects 3-channel input at a multiple of the patch size."""

    def __init__(self, name="facebook/dinov2-small", freeze=True):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(name)
        self.out_dim = int(self.vit.config.hidden_size)
        self.frozen = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.frozen:
            with torch.no_grad():
                out = self.vit(pixel_values=x)
        else:
            out = self.vit(pixel_values=x)
        tok = out.last_hidden_state
        # CLS token concatenated with the mean patch token
        return torch.cat([tok[:, 0], tok[:, 1:].mean(1)], dim=1)



# Head

class PoseNet(nn.Module):
    def __init__(self, backbone, backbone_dim, p_drop=0.2, hidden=512,
                 t_scale=10.0):
        super().__init__()
        self.backbone = backbone
        self.t_scale = t_scale
        self.trunk = nn.Sequential(
            nn.Linear(backbone_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Dropout(p_drop))
        h = hidden // 2
        self.head_t = nn.Linear(h, 3)
        self.head_lv = nn.Linear(h, 3)
        self.head_q = nn.Linear(h, 4)

    def forward(self, x):
        f = self.backbone(x)
        h = self.trunk(f)
        t = self.head_t(h) * self.t_scale
        lv = self.head_lv(h).clamp(-8., 6.)
        q = F.normalize(self.head_q(h), dim=-1)
        return t, lv, q


def build_model(kind="cnn", p_drop=0.2, t_scale=10.0, freeze_vit=True,
                pretrained=True):
    if kind == "cnn":
        b, tag = build_cnn_backbone(pretrained=pretrained, in_ch=1)
        dim = b.out_dim
    elif kind == "vit":
        b = DINOv2Backbone(freeze=freeze_vit)
        dim = b.out_dim * 2          # CLS + mean patch
        tag = "dinov2-small" + ("-frozen" if freeze_vit else "-finetuned")
    else:
        raise ValueError(kind)
    return PoseNet(b, dim, p_drop=p_drop, t_scale=t_scale), tag



# Losses

def translation_nll(t_pred, t_logvar, t_true):
    """Heteroscedastic Gaussian negative log-likelihood, per axis."""
    se = (t_pred - t_true) ** 2
    return (0.5 * (torch.exp(-t_logvar) * se + t_logvar)).mean()


def quaternion_loss(q_pred, q_true):
    """1 - |dot|, which is zero when the rotations agree.

    The absolute value is required because q and -q represent the same
    rotation; without it the network is penalised for a correct answer with the
    opposite sign.
    """
    d = (q_pred * q_true).sum(-1).abs().clamp(max=1.0 - 1e-7)
    return (1.0 - d).mean()


def pose_loss(t, lv, q, t_true, q_true, w_q=5.0):
    return translation_nll(t, lv, t_true) + w_q * quaternion_loss(q, q_true)



# Training

def train(model, X, T, Q, epochs=8, bs=32, lr=1e-3, device="cpu", log=print,
          val=None):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = max(1, len(X) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt = torch.from_numpy(X); Tt = torch.from_numpy(T); Qt = torch.from_numpy(Q)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = nb = 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, tb, qb = Xt[idx].to(device), Tt[idx].to(device), Qt[idx].to(device)
            t, lv, q = model(xb)
            loss = pose_loss(t, lv, q, tb, qb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            tot += loss.item(); nb += 1
        msg = f"  epoch {ep+1:2d}/{epochs}  loss {tot/max(nb,1):+.4f}"
        if val is not None:
            m = evaluate(model, *val, device=device)
            msg += f"   val  t {m['t_err_median']:.3f} m   rot {m['q_err_median']:.2f} deg"
        log(msg)
    return model



# Inference

@torch.no_grad()
def predict(model, X, bs=64, device="cpu"):
    model.to(device).eval()
    ts, lvs, qs = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        t, lv, q = model(xb)
        ts.append(t.cpu().numpy()); lvs.append(lv.cpu().numpy()); qs.append(q.cpu().numpy())
    return np.concatenate(ts), np.concatenate(lvs), np.concatenate(qs)


@torch.no_grad()
def predict_mc(model, X, n_mc=20, bs=64, device="cpu"):
    """Monte Carlo dropout. Returns the translation mean and the uncertainty split.

        aleatoric = mean over passes of the predicted variance
        epistemic = variance over passes of the predicted mean

    The forward passes are run here rather than by calling predict(), because
    predict() puts the whole model in eval mode, which switches dropout off and
    silently collapses the epistemic term to exactly zero.
    """
    model.to(device)
    model.eval()
    n_drop = 0
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()                    # dropout on, batch-norm stays in eval
            n_drop += 1
    if n_drop == 0:
        raise RuntimeError("no Dropout layers found: MC dropout would be meaningless")

    T_all, V_all, Q_all = [], [], []
    for _ in range(n_mc):
        ts, vs, qs = [], [], []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            t, lv, q = model(xb)
            ts.append(t.cpu().numpy()); vs.append(np.exp(lv.cpu().numpy()))
            qs.append(q.cpu().numpy())
        T_all.append(np.concatenate(ts)); V_all.append(np.concatenate(vs))
        Q_all.append(np.concatenate(qs))
    T_all = np.stack(T_all); V_all = np.stack(V_all); Q_all = np.stack(Q_all)

    qm = Q_all.mean(0)
    qm = qm / (np.linalg.norm(qm, axis=1, keepdims=True) + 1e-9)
    return {"t_mean": T_all.mean(0),
            "aleatoric": V_all.mean(0),
            "epistemic": T_all.var(0),
            "q_mean": qm,
            "q_spread": Q_all.std(0).mean(1),
            "n_dropout_layers": n_drop}


def evaluate(model, X, T, Q, device="cpu", bs=64):
    from speedplus_data import quat_angle_deg
    t, lv, q = predict(model, X, bs=bs, device=device)
    te = np.linalg.norm(t - T, axis=1)
    qe = np.array([quat_angle_deg(q[i], Q[i]) for i in range(len(q))])
    rng = np.linalg.norm(T, axis=1)
    return {"t_err_median": float(np.median(te)),
            "t_err_mean": float(te.mean()),
            "q_err_median": float(np.median(qe)),
            "q_err_mean": float(qe.mean()),
            "t_err_rel_median": float(np.median(te / np.maximum(rng, 1e-6))),
            "t_err": te, "q_err": qe, "range": rng}
