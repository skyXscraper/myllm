import keras
from layers import MLP, GQA, RMSNorm
from keras import layers
from config import MyllmConfig

class MyllmDecoder(layers.Layer):
    def __init__(self, config: MyllmConfig, **kwargs):
        super().__init__(**kwargs)
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQA(
            d_model=config.d_model, 
            n_heads=config.n_heads, 
            n_kv_heads=config.n_kv_heads, 
            seq_len=config.max_seq_len
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = MLP(
            d_model=config.d_model, 
            intermediate_dim=config.intermediate_dim
        )
    
    def call(self, x):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out
    
def build_myllm(config: MyllmConfig):
    inputs = layers.Input(shape=(config.max_seq_len,), dtype="int32")
    x = layers.Embedding(config.vocab_size, config.d_model)(inputs)

    for i in range(config.num_layers):
        x = MyllmDecoder(config, name=f"decoder_layer_{i}")(x)

    x = RMSNorm(config.d_model)(x)
    outputs = layers.Dense(config.vocab_size, use_bias=False, name="lm_head")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MyLLM")
    return model