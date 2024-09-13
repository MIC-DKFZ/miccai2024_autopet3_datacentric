import typer
from autopet3.datacentric.dataloader import AutoPETDataModule
from autopet3.datacentric.logger import get_logger
from autopet3.datacentric.setup import setup
from autopet3.datacentric.utils import result_parser
from autopet3.fixed.dynunet import NNUnet
from autopet3.fixed.utils import SaveFileToLoggerDirCallback
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner import Tuner
import subprocess

app = typer.Typer()


def build(config: str):
    setup()
    config = OmegaConf.load(config)
    print(config)
    if hasattr(config.trainer, "deterministic"):
        if config.trainer.deterministic:
            seed = 42
            if hasattr(config.model, "seed"):
                seed = config.model.seed
            seed_everything(seed, workers=True)

    config.data.num_workers_train = 12
    config.data.num_workers_val = 12

    datamodule = AutoPETDataModule(**config.data)
    logger = get_logger(config)

    if config.model.pretrained:
        print(config.model.ckpt_path)
        net = NNUnet.load_from_checkpoint(config.model.ckpt_path)
    else:
        net = NNUnet(config.model.lr)
    return net, datamodule, config, logger

@app.command()
def train(config: str = "config/config.yml", debug: bool = False):
    config_file = config
    net, datamodule, config, logger = build(config)

    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True)

    config_callback = SaveFileToLoggerDirCallback(config_file)
    checkpoint_callback = ModelCheckpoint(filename="{epoch}-{step}", monitor="val/loss", mode="min", save_last=True)
    trainer = Trainer(callbacks=[lr_monitor, config_callback, checkpoint_callback], logger=logger, **config.trainer)

    if hasattr(config.model, "resume"):
        if config.model.resume:
            trainer.fit(net, datamodule, ckpt_path=config.model.ckpt_path)
        else:
            trainer.fit(net, datamodule)
    else:
        trainer.fit(net, datamodule)


@app.command()
def find_lr(config: str = "config/config.yml"):
    net, datamodule, config, logger = build(config)
    trainer = Trainer(logger=logger, **config.trainer)

    tuner = Tuner(trainer)
    tuner.lr_find(net, datamodule=datamodule)


@app.command()
def find_batchsize(config: str = "config/config.yml"):
    net, datamodule, config, logger = build(config)

    config.trainer.devices = 1
    config.trainer.num_nodes = 1

    trainer = Trainer(logger=logger, **config.trainer)

    tuner = Tuner(trainer)
    tuner.scale_batch_size(net, datamodule=datamodule, mode="binsearch")


@app.command()
def test(config: str = "config/config.yml"):
    net, datamodule, config, logger = build(config)

    config.trainer.devices = 1
    config.trainer.num_nodes = 1

    trainer = Trainer(logger=logger, **config.trainer)
    trainer.test(net, datamodule=datamodule)
    print(
        "Be aware that the metrics are calculated on the network target spacing and might not be 100% accurate. You\n"
        "will need to resample the predictions for the final evaluation to match the input spacing and calculate \n"
        "the metrics based on this resolution before aggregating."
    )

    # save results
    try:
        result_parser(net, datamodule, trainer)
    except Exception as e:
        print(
            "Could not save results to json file. If you do not use a MONAI dict dataset please change the test code "
            "parser to match your dataset in autopet3/datacentric/utils.py."
        )
        print(e)


if __name__ == "__main__":
    app()
