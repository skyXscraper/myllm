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

class GQA(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, seq_len: int, **kwargs):
        super().__init__(**kwargs)

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_queris_per_kv = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len

        self.q_proj = layers.Dense(n_heads * self.head_dim, use_bias=False)
        self.k_proj = layers.Dense(n_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = layers.Dense(n_kv_heads * self.head_dim, use_bias=False)
        self.o_proj = layers.Dense(d_model, use_bias=False)

    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        q = keras.ops.reshape(self.q_proj(x), (batch_size, self.seq_len, self.n_heads, self.head_dim))
        k = keras.ops.reshape(self.k_proj(x), (batch_size, self.seq_len, self.n_kv_heads, self.head_dim))
        v = keras.ops.reshape(self.v_proj(x), (batch_size, self.seq_len, self.n_kv_heads, self.head_dim))
        q, k = rope(q, k, self.seq_len, self.head_dim)

        if self.num_queries_per_kv > 1:
            k = keras.ops.repeat(k, self.num_queries_per_kv, axis=2)
            v = keras.ops.repeat(v, self.num_queries_per_kv, axis=2)

        q = keras.ops.transpose(q, (0, 2, 1 , 3))
        k = keras.ops.transpose(k, (0, 2, 1 , 3))
        v = keras.ops.transpose(v, (0, 2, 1 , 3))

        mask = 1.0 - keras.ops.tri(self.seq_len, self.seq_len, k=0)

        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / keras.ops.sqrt(self.head_dim)
        scores = scores + (mask[None, None, :, :] * -1e9)

        att_probs = keras.ops.softmax(scores, axis=-1)
        output = keras.ops.matmul(att_probs, v)
        output = keras.ops.transpose(output, (0, 2, 1, 3))
        output = keras.ops.reshape(output, (batch_size, self.seq_len, -1))

        return self.o_proj(output)