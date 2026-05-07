import keras
from keras import layers
import numpy as np

keras.config.set_floatx('float32')

class RMSNorm(layers.Layer):
    def __init__(self, dim: int, esp: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.esp = esp
        self.scale = self.add_weight(
            name="scale", 
            shape=(dim, ), 
            initializer="ones",
            trainable=True
        )

    def call(self, x):
        var = keras.ops.mean(keras.ops.square(x), axis=-1, keepdims=True)
        return x * keras.ops.rsqrt(var + self.esp) * self.scale
    
def rope(q, k, seq_len: int, head_dim: int):
    inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype='float32') / head_dim))
    t = np.arange(seq_len, dtype='float32')
    freqs = np.outer(t, inv_freq)
    emb = np.concatentate([freqs, freqs], axis=-1)
    cos = keras.ops.cast(np.cos(emb), dtype='float32')[None, :, None, :]
    sin = keras.ops.cast(np.sin(emb), dtype='float32')[None, :, None, :]
    def rotate_half(x):
        half_size = head_dim // 2
        x1 = x[..., :half_size]
        x2 = x[..., half_size:]
        return keras.ops.concatenate([-x2, x1], axis=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MLP(layers.Layer):
    def __init__(self, d_model: int, intermediate_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gate_proj = layers.Dense(intermediate_dim, use_bias=False)
        self.up_proj = layers.Dense(intermediate_dim, use_bias=False)
        self.down_proj = layers.Dense(d_model, use_bias=False)

    def call(self, x):
        gate = keras.ops.silu(self.gate_proj(x))
        activated = gate * self.up_proj(x)
        return self.down_proj(activated)

