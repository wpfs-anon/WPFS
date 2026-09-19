"""The purpose-built corrector: chunk positions query pi0's frozen tokens.

Same definition the trainers use, lifted out so the rollout driver can load a
checkpoint without importing a trainer.  Dimensions come from the checkpoint, so
a checkpoint carries everything needed to rebuild its network.

Two memory layouts exist and a checkpoint says which one it wants:

  gemma_layers == 0   54_train_net.py.  Memory is the vision tower's output for
                      two cameras, pi0's token embedding for the instruction,
                      and the proprioceptive state -- three streams, three type
                      embeddings.

  gemma_layers  > 0   train_net_student.py.  Memory is pi0's prefix after that many
                      Gemma layers, image and language already fused, so it is
                      one stream plus state.  ``lng_in`` exists in the state dict
                      but is never called.

Feeding a network the layout it was not trained on produces plausible numbers
that mean nothing, so ``NetCorrector`` reads ``gemma_layers`` from the file and
asserts the token count it built matches what the network expects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class Block(nn.Module):
    def __init__(self, d, h, mult):
        super().__init__()
        self.h = h
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o1 = nn.Linear(d, d)
        self.q2 = nn.Linear(d, d)
        self.kv2 = nn.Linear(d, 2 * d)
        self.o2 = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                                 nn.Linear(mult * d, d))

    def _split(self, t, B, L):
        return t.view(B, L, self.h, -1).transpose(1, 2)

    def forward(self, x, mem):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        a = Fn.scaled_dot_product_attention(self._split(q, B, L), self._split(k, B, L),
                                            self._split(v, B, L))
        x = x + self.o1(a.transpose(1, 2).reshape(B, L, d))
        M = mem.shape[1]
        q = self._split(self.q2(self.n2(x)), B, L)
        k, v = self.kv2(mem).chunk(2, dim=-1)
        a = Fn.scaled_dot_product_attention(q, self._split(k, B, M), self._split(v, B, M))
        x = x + self.o2(a.transpose(1, 2).reshape(B, L, d))
        return x + self.ffn(self.n3(x))


class StudentNet(nn.Module):
    def __init__(self, d, blocks, heads, mult, c_tok, s_dim, H, D, feat_dim=0,
                 plan_dim=0, use_age=False):
        super().__init__()
        self.H, self.D = H, D
        self.img_in = nn.Linear(c_tok, d)
        self.lng_in = nn.Linear(c_tok, d)
        self.st_in = nn.Linear(s_dim, d)
        self.mem_type = nn.Parameter(torch.zeros(3, d))
        self.chunk_in = nn.Linear(D, d)
        self.pos = nn.Parameter(torch.zeros(H, d))
        self.tau_in = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, mult) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D))
        # feat_dim: predict the teacher's last hidden state and let the
        # teacher's own frozen projection read the velocity off it.  Buffers, so
        # they travel in the checkpoint and never train.
        self.feat_dim = feat_dim
        if feat_dim:
            self.feat_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, feat_dim))
            self.register_buffer("head_w", torch.zeros(D, feat_dim))
            self.register_buffer("head_b", torch.zeros(D))
        # plan context: the teacher's state when this plan was made, as extra
        # memory tokens, and how many steps ago that was
        self.plan_dim, self.use_age = plan_dim, use_age
        if plan_dim:
            self.plan_in = nn.Sequential(nn.LayerNorm(plan_dim), nn.Linear(plan_dim, d))
            self.plan_type = nn.Parameter(torch.zeros(d))
        if use_age:
            self.age_in = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d))

    def memory(self, img, lng, state):
        """gemma_layers == 0 layout: image, language and state as three streams."""
        return torch.cat([self.img_in(img) + self.mem_type[0],
                          self.lng_in(lng) + self.mem_type[1],
                          self.st_in(state)[:, None] + self.mem_type[2]], dim=1)

    def memory_fused(self, tok, state, plan=None):
        """gemma_layers > 0 layout: one fused prefix stream, plus state, plus
        the plan's own tokens when the checkpoint was trained with them."""
        m = [self.img_in(tok) + self.mem_type[0],
             self.st_in(state)[:, None] + self.mem_type[2]]
        if self.plan_dim and plan is not None:
            # the loop hands one observation (T, plan_dim); the trainer a batch
            m.append(self.plan_in(plan[None] if plan.dim() == 2 else plan) + self.plan_type)
        return torch.cat(m, dim=1)

    def hidden(self, x, tau, mem, age=None):
        """The block stack's output, before the action head.  This is the
        representation feature distillation matches to the teacher's own."""
        half = torch.exp(torch.linspace(0, -9, 128, device=x.device))
        f = tau[:, None] * half[None] * 1000.0
        h = self.chunk_in(x) + self.pos[None] \
            + self.tau_in(torch.cat([f.sin(), f.cos()], -1))[:, None]
        if self.use_age and age is not None:
            g = (age.float() / 50.0)[:, None] * half[None] * 1000.0
            h = h + self.age_in(torch.cat([g.sin(), g.cos()], -1))[:, None]
        for blk in self.blocks:
            h = blk(h, mem)
        return h

    def forward_feat(self, x, tau, mem, age=None):
        """(velocity, predicted teacher feature) through the frozen head."""
        f = self.feat_out(self.hidden(x, tau, mem, age))
        return torch.nn.functional.linear(f, self.head_w, self.head_b), f

    def forward(self, x, tau, mem, age=None):
        if self.feat_dim:
            return self.forward_feat(x, tau, mem, age)[0]
        return self.out(self.hidden(x, tau, mem, age))


def prefix_tokens(model, be, images, masks, lang, s_dim, device, n_layers):
    """pi0's prefix after n_layers, sliced to the tokens the net reads.

    This is the ONE definition; train_net_student.py imports it rather than keeping
    a copy, because the two did diverge and the eval then built a memory the
    network was never trained on.  The third camera slot is
    zeros with mask False in LIBERO and its tokens are dropped; the teacher's
    plan path still runs all three and is untouched.
    """
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from sentry.models.openpi_adapter import _Namespace
    obs = _Namespace(images=images, image_masks=masks,
                     state=torch.zeros(lang.shape[0], s_dim, device=device),
                     tokenized_prompt=lang, tokenized_prompt_mask=(lang != 0).bool(),
                     token_ar_mask=None, token_loss_mask=None)
    im, imm, lg, lgm, _ = model._preprocess_observation(obs, train=False)
    if n_layers == 0:
        # Vision tower output plus pi0's own token embedding for the
        # instruction.  Dropping the language here is what produced a 513-token
        # memory against a 561-token checkpoint.
        pwe = model.paligemma_with_expert
        img = torch.cat([pwe.embed_image(im[i]) for i in (0, 1)], dim=1)
        emb = (pwe.embed_language_tokens if hasattr(pwe, "embed_language_tokens")
               else pwe.paligemma.language_model.model.embed_tokens)
        return torch.cat([img, emb(lang)], dim=1)
    with be._truncated(n_layers, be.L_B):
        e, pm, am = model.embed_prefix(im, imm, lg, lgm)
        be._vlm.config._attn_implementation = "eager"
        out, _ = be._pwe.forward(
            attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(pm, am)),
            position_ids=torch.cumsum(pm, dim=1) - 1,
            past_key_values=None, inputs_embeds=[e, None], use_cache=True)
    hs = out[0]
    per_cam = (hs.shape[1] - lang.shape[1]) // 3
    return torch.cat([hs[:, :2 * per_cam], hs[:, 3 * per_cam:]], dim=1)


class NetCorrector:
    """Wraps the net with whichever pi0 frontend its checkpoint was trained on."""

    def __init__(self, path, model, be, H, D, device="cuda"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = StudentNet(ck["d_model"], ck["blocks"], ck["heads"],
                              ck["ffn_mult"], ck["c_tok"], ck["s_dim"],
                              H, D, feat_dim=int(ck.get("feat_dim", 0)),
                              plan_dim=int(ck.get("plan_dim", 0)),
                              use_age=bool(ck.get("use_age", False))).to(device)
        self.net.load_state_dict(ck["net"], strict=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.ck, self.device, self.model, self.be = ck, device, model, be
        self.n_layers = int(ck.get("gemma_layers", 0))
        # train_net_student.py fuses image and language into one memory stream and
        # records mem_tokens; 54_train_net.py keeps three streams and does not.
        # gemma_layers alone does not distinguish them: a zero-layer 56_ run is
        # fused too, and feeding it the split layout is silent nonsense.
        self.fused = "mem_tokens" in ck
        self.s_dim = int(ck["s_dim"])
        self.pwe = model.paligemma_with_expert
        if hasattr(self.pwe, "embed_language_tokens"):
            self.emb = self.pwe.embed_language_tokens
        else:
            self.emb = self.pwe.paligemma.language_model.model.embed_tokens
        self._checked = False

    def describe(self):
        c = self.ck
        n = sum(p.numel() for p in self.net.parameters())
        return (f"net {c['d_model']}x{c['blocks']} ({n/1e6:.1f}M params), "
                f"gemma_layers {self.n_layers}, step {c['step']}, "
                f"rel L2 {c['rel']:.3f}, spread {c['spread']:.2f}x")

    def _images(self, obs):
        dev = self.device
        img = obs.images.to(dev, torch.float32)
        ims, msk = {}, {}
        for j, key in enumerate(self.be.image_keys):
            ims[key] = img[j][None] if j < img.shape[0] else torch.zeros_like(img[0])[None]
            msk[key] = torch.tensor([j < img.shape[0]], device=dev)
        return ims, msk

    @torch.no_grad()
    def memory(self, obs, plan=None):
        dev = self.device
        st = obs.state.to(dev, torch.float32)[None]
        if not self.fused:
            tok = torch.cat([self.pwe.embed_image(obs.images[i:i + 1].to(dev, torch.float32))
                             for i in range(obs.images.shape[0])], dim=1).float()
            lng = self.emb(obs.language.to(dev)[None]).float()
            mem = self.net.memory(tok, lng, st)
        else:
            ims, msk = self._images(obs)
            tok = prefix_tokens(self.model, self.be, ims, msk,
                                obs.language.to(dev)[None], self.s_dim,
                                dev, self.n_layers).float()
            mem = self.net.memory_fused(tok, st, plan)
        if not self._checked:
            # A layout mismatch is silent otherwise: the shapes broadcast and the
            # network happily attends to the wrong thing.
            want = self.ck.get("mem_tokens")
            if want is not None and mem.shape[1] != int(want):
                raise SystemExit(
                    f"memory has {mem.shape[1]} tokens but the checkpoint was "
                    f"trained on {want}; gemma_layers={self.n_layers} is wrong "
                    "for this file")
            print(f"  net memory: {mem.shape[1]} tokens x {mem.shape[2]} "
                  f"(gemma_layers {self.n_layers})")
            self._checked = True
        return mem

    @torch.no_grad()
    def velocity(self, x, tau, mem, age=None):
        K = x.shape[0]
        ag = (torch.full((K,), float(age), device=self.device)
              if (age is not None and self.net.use_age) else None)
        return self.net(x.to(self.device).float(), tau.to(self.device).float(),
                        mem.expand(K, -1, -1), ag)
