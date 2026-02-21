import os
from config.config import DATA_PATH
from data.dataset_generator import generate_dataset
from training.train import train_model


def main():
    # Generate dataset if it doesn't exist
    if not os.path.exists(DATA_PATH):
        print("Generating dataset...")
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        print("Dataset created at:", DATA_PATH)

    print("Starting training pipeline...")
    model = train_model(DATA_PATH)


if __name__ == "__main__":
    main()