import torch
import torch.nn as nn
import torch.nn.functional as Fn


class Block(nn.Module):
    def __init__(self, d, h, mult=4):
        super().__init__()
        self.h = h
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o1 = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.q2, self.kv2, self.o2 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def _s(self, t, B, L):
        return t.view(B, L, self.h, -1).transpose(1, 2)

    def forward(self, x, mem):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        o = Fn.scaled_dot_product_attention(self._s(q, B, L), self._s(k, B, L), self._s(v, B, L))
        x = x + self.o1(o.transpose(1, 2).reshape(B, L, d))
        M = mem.shape[1]
        k, v = self.kv2(mem).chunk(2, -1)
        o = Fn.scaled_dot_product_attention(self._s(self.q2(self.n2(x)), B, L), self._s(k, B, M), self._s(v, B, M))
        x = x + self.o2(o.transpose(1, 2).reshape(B, L, d))
        return x + self.ffn(self.n3(x))


def sincos(v, n=128):
    half = torch.exp(torch.linspace(0, -9, n, device=v.device))
    f = v[:, None].float() * half[None] * 1000.0
    return torch.cat([f.sin(), f.cos()], -1)


class DboftStudent(nn.Module):
    def __init__(self, d, blocks, heads, c_img, c_txt, H, D):
        super().__init__()
        self.cfg = dict(d=d, blocks=blocks, heads=heads, c_img=c_img, c_txt=c_txt, H=H, D=D)
        self.img_in, self.txt_in, self.anc_in = nn.Linear(c_img, d), nn.Linear(c_txt, d), nn.Linear(D, d)
        self.mem_type = nn.Parameter(torch.zeros(3, d))
        self.anc_pos, self.pos = nn.Parameter(torch.zeros(H, d)), nn.Parameter(torch.zeros(H, d))
        self.chunk_in = nn.Linear(D, d)
        self.t_in = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d))
        self.age_in = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D))

    def memory(self, clip, txt, anchor):
        return torch.cat([self.img_in(clip) + self.mem_type[0], self.txt_in(txt) + self.mem_type[1],
                          self.anc_in(anchor) + self.anc_pos[None] + self.mem_type[2]], 1)

    def forward(self, x, t, mem, age):
        h = self.chunk_in(x) + self.pos[None] + self.t_in(sincos(t / 1000.0))[:, None] \
            + self.age_in(sincos(age / 50.0))[:, None]
        for b in self.blocks:
            h = b(h, mem)
        return self.out(h)


