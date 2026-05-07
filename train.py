# train.py
import os
import keras
from config import MyllmConfig
from models.myllm import build_myllm
from data.dataset import get_dataset

def run_training():
    config = MyllmConfig()
    print(f"Initializing MyLLM Training Pipeline...")

    dataset, tokenizer = get_dataset(config)
    print("Streaming dataset initialized. Tokenizer loaded.")

    model = build_myllm(config)
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    checkpoint_path = "checkpoints/myllm_120m.keras"
    os.makedirs("checkpoints", exist_ok=True)
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="loss",
        verbose=1
    )

    print(f"Starting training for {config.epochs} epoch(s) on CPU...")
    model.fit(
        dataset,
        epochs=config.epochs,
        callbacks=[checkpoint_callback]
    )

    final_model_path = "models/myllm_120m_final.keras"
    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")

if __name__ == "__main__":
    run_training()