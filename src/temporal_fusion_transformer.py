import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch


from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE

from sklearn.metrics import mean_squared_error, d2_tweedie_score

from models.tft import TemporalFusionTransformer


def seed_everything(seed=30):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_tft(data_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        raise Exception("A GPU is required to train this model.")

    seed_everything()

    df = pd.read_csv(data_path + "/clean_dataset.csv")

    df.drop(
        columns=["PRECIO", "IMPORTELINEA", "PRODUCTO", "PURCHASING_REGION"],
        inplace=True,
    )

    df["CODIGO_NUM"] = df["CODIGO_NUM"].astype(str)
    df["PURCHASING_DEPARTMENT"] = df["PURCHASING_DEPARTMENT"].astype(str)
    df["PURCHASING_HOSPITAL"] = df["PURCHASING_HOSPITAL"].astype(str)
    df["YEAR"] = df["YEAR"].astype(str)
    df["MONTH"] = df["MONTH"].astype(str)

    train = df[df["YEAR"] != 2023]
    test = df[df["YEAR"] == 2023]

    max_prediction_length = 365  # We will predict the entire 2023 year
    max_encoder_length = train["time_idx"].nunique()

    training = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="CANTIDADCOMPRA",
        group_ids=["CODIGO_NUM", "PURCHASING_DEPARTMENT", "PURCHASING_HOSPITAL"],
        min_encoder_length=max_prediction_length // 4,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[
            "CODIGO_NUM",
            "PURCHASING_DEPARTMENT",
            "PURCHASING_HOSPITAL",
        ],
        time_varying_known_categoricals=["YEAR", "MONTH"],
        time_varying_known_reals=["time_idx", "DAYOFMONTH", "DAYOFYEAR"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["CANTIDADCOMPRA"],
        categorical_encoders={"TIPOCOMPRA": NaNLabelEncoder(add_nan=True)},
        # lags={'CANTIDADCOMPRA': [30, 90, 365]},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, train, predict=True, stop_randomization=True
    )

    batch_size = 64  # set this between 32 to 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=3
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=3
    )

    PATIENCE = 30
    MAX_EPOCHS = 120
    LEARNING_RATE = 0.03

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        min_delta=1e-2,
        patience=PATIENCE,
        verbose=False,
        mode="min",
    )
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.25,
        limit_train_batches=10,  # coment in for training, running valiation every 30 batches
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        lstm_layers=2,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=8,
        output_size=1,  # 7 quantiles by default
        loss=SMAPE(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )

    tft.to(DEVICE)
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).to(DEVICE)
    predictions = best_tft.predict(val_dataloader, mode="prediction")
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    sm = SMAPE()
    smape_loss = sm.loss(actuals, predictions).mean(axis=1)
    tweedie = d2_tweedie_score(actuals, predictions)
    test_loss = mean_squared_error(actuals, predictions, squared=False)

    return smape_loss, tweedie, test_loss
