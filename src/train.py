import argparse
import warnings

# Ignore warnings from several libraries, as they are not relevant for the task
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Relative path to training data"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model to train ('boltzmann' or 'tft')"
    )

    args = parser.parse_args()

    data_path = args.data_path
    model = args.model

    if model == "boltzmann":
        from boltzmann_ensemble import train_boltzmann_ensemble

        train_boltzmann_ensemble(data_path)
    elif model == "tft":
        from temporal_fusion_transformer import train_tft

        train_tft(data_path)
    else:
        raise ValueError(
            f"Model {model} not implemented, choose between a Boltzmann Ensemble ('--model boltzmann') and a Temporal Fusion Transformer ('--model tft')"
        )
