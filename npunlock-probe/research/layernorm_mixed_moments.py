"""Independent CPU check of an eight-lane cascaded Welford ordering, not a runtime fallback."""
import numpy as np


def moments(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    f = np.float32
    def fused(a, b, c):
        return (np.asarray(a, dtype=np.float64)*np.asarray(b, dtype=np.float64)+np.asarray(c, dtype=np.float64)).astype(np.float32)
    def merge(n, m, v, count, mean, variance, vector=True):
        ratio = f(count)/f(n+count) if n+count else f(0)
        delta = mean-m
        if vector:
            return n+count, m+ratio*delta, fused(delta*f(n), ratio*delta, v+variance)
        return n+count, f(m+ratio*delta), f(v+f(variance+f(f(f(delta*delta)*ratio)*f(n))))
    vectors = len(values)//16
    chunks = (vectors+15)//16
    depth = (chunks-1).bit_length() if chunks else 0
    stack = [(0, np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)) for _ in range(max(1, depth))]
    for chunk in range(chunks):
        count = min(16, vectors-chunk*16)
        for half in (0, 1):
            mean, variance = np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)
            for j in range(count):
                x = values[chunk*256+j*16+half*8:chunk*256+j*16+half*8+8]
                delta = x-mean
                mean = fused(delta, f(1)/f(j+1), mean)
                variance = fused(delta, x-mean, variance)
            stack[0] = merge(*stack[0], count, mean, variance)
        mask, level = chunk+1, 1
        while level < depth and not mask & 1:
            stack[level] = merge(*stack[level], *stack[level-1])
            stack[level-1] = (0, np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
            mask >>= 1
            level += 1
    for level in range(1, depth):
        stack[0] = merge(*stack[0], *stack[level])
    count, mean, variance = 0, f(0), f(0)
    for x in values[vectors*16:]:
        delta = x-mean
        count += 1
        mean = f(mean+f(delta/f(count)))
        variance = f(variance+f(delta*f(x-mean)))
    for lane in range(8):
        count, mean, variance = merge(count, mean, variance, vectors*2, stack[0][1][lane], stack[0][2][lane], False)
    return mean, f(variance/f(len(values)))


if __name__ == '__main__':
    import torch
    from layernorm_mixed_diagnose import sample
    torch.set_num_threads(1)
    x,w,b,_ = sample()
    _, m, a = torch.native_layer_norm(x, (4096,), w, b, 1e-5)
    mean, variance = moments(x.numpy())
    inverse = np.float32(1)/np.sqrt(variance+np.float32(1e-5))
    print('ordered mean/inverse bits', mean.view(np.uint32), inverse.view(np.uint32))
    print('CPU mean/inverse bits', m.numpy().view(np.uint32), a.numpy().view(np.uint32))
    assert mean.view(np.uint32) == m.numpy().view(np.uint32).item()
    assert inverse.view(np.uint32) == a.numpy().view(np.uint32).item()
