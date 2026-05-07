import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MyllmConfig

def get_dataset(config: MyllmConfig):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "cosmopedia-v2", 
        split="train", 
        streaming=True
    )

    def process_and_tokenize(sample):
        text = sample["text"]
        encodings = tokenizer(
            text, 
            truncation=True,
            max_length=config.seq_len + 1, 
            padding="max_length", 
            return_tensors="np"
        )
        ids = encodings["input_ids"][0].astype('int32')
        return ids[:-1], ids[1:]
    
    def generator():
        for sample in raw_dataset.take(10000):
            yield process_and_tokenize(sample)

    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=(
            tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32), 
            tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)
        )
    )
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer
