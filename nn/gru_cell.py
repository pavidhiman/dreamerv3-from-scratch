import numpy as np
from nn.linear import Linear
from nn.tensor import Tensor, cat, _xp, GPU

INFERENCE_MODE = False

_fused_gru_kernel = None
if GPU:
    import cupy as cp
    _fused_gru_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void fused_gru_elementwise(
        const float* rz_out,    // (batch, 2*H) = concat of reset and update pre-activations
        const float* h_prev,    // (batch, H)
        const float* x_data,    // (batch, input_size)
        const float* Wc,        // (input_size+H, H)
        const float* bc,        // (H,)
        float* h_new,           // (batch, H) output
        float* r_out,           // (batch, H) reset gate
        float* z_out,           // (batch, H) update gate
        float* cand_out,        // (batch, H) candidate
        int batch, int input_size, int H
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * H;
        if (idx >= total) return;

        int b = idx / H;
        int j = idx % H;

        float r_pre = rz_out[b * (2*H) + j];
        float z_pre = rz_out[b * (2*H) + H + j];
        float r = 1.0f / (1.0f + expf(-r_pre));
        float z = 1.0f / (1.0f + expf(-z_pre));
        float h_p = h_prev[b * H + j];

        float cand_pre = bc[j];
        for (int k = 0; k < input_size; k++) {
            cand_pre += x_data[b * input_size + k] * Wc[k * H + j];
        }
        for (int k = 0; k < H; k++) {
            float rh = (k == j) ? r * h_p : (1.0f / (1.0f + expf(-rz_out[b*(2*H) + k]))) * h_prev[b*H + k];
            cand_pre += rh * Wc[(input_size + k) * H + j];
        }
        float cand = tanhf(cand_pre);

        float h_out = z * h_p + (1.0f - z) * cand;

        h_new[idx] = h_out;
        r_out[idx] = r;
        z_out[idx] = z;
        cand_out[idx] = cand;
    }
    ''', 'fused_gru_elementwise')


class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = Linear(input_size + hidden_size, hidden_size)
        self.update_gate = Linear(input_size + hidden_size, hidden_size)
        self.candidate = Linear(input_size + hidden_size, hidden_size)

    def __call__(self, x, h):
        if h is None: h = Tensor(_xp.zeros((x.shape[0], self.hidden_size), dtype=_xp.float32))
        H = self.hidden_size
        batch = x.shape[0]

        if _fused_gru_kernel is not None and INFERENCE_MODE:
            combined = _xp.concatenate([x.data, h.data], axis=1)
            W_rz = _xp.concatenate([self.reset_gate.weight.data, self.update_gate.weight.data], axis=1)
            b_rz = _xp.concatenate([self.reset_gate.bias.data, self.update_gate.bias.data])
            rz_pre = _xp.matmul(combined, W_rz) + b_rz

            h_new = _xp.empty((batch, H), dtype=_xp.float32)
            r_buf = _xp.empty((batch, H), dtype=_xp.float32)
            z_buf = _xp.empty((batch, H), dtype=_xp.float32)
            c_buf = _xp.empty((batch, H), dtype=_xp.float32)

            threads = 256
            blocks = (batch * H + threads - 1) // threads
            _fused_gru_kernel((blocks,), (threads,), (
                rz_pre, h.data, x.data,
                self.candidate.weight.data, self.candidate.bias.data,
                h_new, r_buf, z_buf, c_buf,
                batch, self.input_size, H
            ))
            return Tensor(h_new)

        combined = cat([x, h], axis=1)
        r = self.reset_gate(combined).sigmoid()
        z = self.update_gate(combined).sigmoid()
        candidate = self.candidate(cat([x, r * h], axis=1)).tanh()
        return z * h + (1 - z) * candidate

    def parameters(self):
        p = []
        for l in [self.reset_gate, self.update_gate, self.candidate]: p.extend(l.parameters())
        return p
