from aitextgen import aitextgen
from aitextgen.utils import build_gpt2_config
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
import os


file_name = "fen.txt"
model_dir = "trained_model"
config_file = os.path.join(model_dir, "config.json")
pytorch_model_file = os.path.join(model_dir, "pytorch_model.bin")
vocab_file = os.path.join(model_dir, "aitextgen-vocab.json")
merges_file = os.path.join(model_dir, "aitextgen-merges.txt")
dataset_cache_file = os.path.join(model_dir, "dataset_cache.tar.gz")
max_length = 100
vocab_size = 10000


def train():
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # train tokenizer if necessary
    if not os.path.exists(vocab_file):
        print("training tokenizer, please wait...")
        train_tokenizer(file_name, save_path=model_dir,
                        vocab_size=vocab_size, serialize=False)

    if os.path.exists(dataset_cache_file):  # use cache
        data = TokenDataset(
            dataset_cache_file,
            vocab_file=vocab_file,
            merges_file=merges_file,
            block_size=max_length,
            from_cache=True,
        )
    else:  # or create token cache if necessary
        data = TokenDataset(
            file_name,
            vocab_file=vocab_file,
            merges_file=merges_file,
            block_size=max_length,
            line_by_line=True,
            save_cache=True,
            cache_destination=dataset_cache_file
        )

    if not os.path.exists(pytorch_model_file):
        config = build_gpt2_config(
            vocab_size=vocab_size,
            max_length=max_length,
            dropout=0.0,
            n_embd=512,
            n_head=16,
            n_layer=16,
        )

        ai = aitextgen(
            config=config, vocab_file=vocab_file, merges_file=merges_file, to_gpu=False
        )  # Change to_gpu
    else:
        ai = aitextgen(
            model=pytorch_model_file,
            config=config_file,
            vocab_file=vocab_file,
            merges_file=merges_file,
            to_gpu=True
        )

    ai.train(
        data,
        num_steps=150000,
        generate_every=1000,
        save_every=1000,
        learning_rate=1e-4,
        batch_size=16,
        num_workers=4,
    )


if __name__ == '__main__':
    train()
