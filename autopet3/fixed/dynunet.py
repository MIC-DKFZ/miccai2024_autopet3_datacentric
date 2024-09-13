import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import ConfusionMatrixMetric, DiceMetric
from monai.networks.nets import DynUNet

# from autopet3.datacentric.utils import PolyLRScheduler
from autopet3.fixed.evaluation import AutoPETMetricAggregator


class NNUnet(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, sw_batch_size: int = 2):
        """Initialize the class with the given learning rate and sliding window batch size.
        Args:
            learning_rate (float): The learning rate for the model.
            sw_batch_size (int): The batch size for sliding window inference.
        Returns:
            None

        """
        super().__init__()
        self.scheduler = True
        self.scheduler_type = "polylr"
        self.scheduler_steps = None
        self.deep_supervision = True
        self.kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        self.strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
        self.patch_size = (128, 160, 112)
        self.sw_mode = "constant"
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = 0.5

        self.backbone = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            kernel_size=self.kernels,
            strides=self.strides,
            upsample_kernel_size=self.strides[1:],
            norm_name="instance",
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            deep_supervision=self.deep_supervision,
            deep_supr_num=3,
            res_block=True,
        )

        self.learning_rate = learning_rate
        self.steps = None

        # formulated as DiceBCE and batch is True according to 3d_fullres plans
        self.loss_fn = DiceCELoss(sigmoid=True, batch=True, include_background=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.confusion = ConfusionMatrixMetric(reduction="mean", metric_name="f1 score")
        self.test_aggregator = AutoPETMetricAggregator()

        self.train_loss = []
        self.val_loss = []
        self.val_dice = []

    def forward(self, volume):
        # return prediction
        pred = self.sliding_window_inference(volume)
        return torch.ge(torch.sigmoid(pred), 0.5)

    def forward_sigmoid(self, volume):
        # return sigmoid
        pred = self.sliding_window_inference(volume)
        return torch.sigmoid(pred)

    def forward_logits(self, volume):
        # return logits
        pred = self.sliding_window_inference(volume)
        return pred

    def compute_loss(self, prediction, label):
        if self.deep_supervision:
            loss, weights = 0.0, 0.0
            for i in range(prediction.shape[1]):
                loss += self.loss_fn(prediction[:, i], label) * 0.5**i
                weights += 0.5**i
            return loss / weights
        return self.loss_fn(prediction, label)

    def training_step(self, batch, batch_idx):
        volume, label = batch
        prediction = self.backbone(volume)
        loss = self.compute_loss(prediction, label)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        volume, label = batch
        prediction = self.sliding_window_inference(volume)
        loss = self.loss_fn(prediction, label)

        prediction = torch.ge(torch.sigmoid(prediction), 0.5)
        self.dice_metric(y_pred=prediction, y=label)

        self.confusion(y_pred=prediction, y=label)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self):
        mean_val_dice = self.dice_metric.aggregate().item()
        self.val_dice.append(mean_val_dice)
        self.dice_metric.reset()
        self.log(
            "val/dice",
            mean_val_dice,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        mean_fp = self.confusion.aggregate()[0].item()
        self.confusion.reset()
        self.log(
            "val/f1",
            mean_fp,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"val/dice": mean_val_dice, "val/f1": mean_fp}

    def test_step(self, batch, batch_idx):
        volume, label = batch
        pred = self.forward(volume)
        assert volume.shape[0] == 1, "Test step just works for batch size 1"
        self.log_dict(self.test_aggregator.update(pred.cpu().numpy(), label.cpu().numpy()))

    def on_test_end(self):
        results = self.test_aggregator.compute()
        print(results)
        return results

    def set_scheduler_steps(self, steps):
        # fallback if trainer is not used
        self.steps = steps

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.SGD(
            self.backbone.parameters(), self.learning_rate, weight_decay=3e-5, momentum=0.99, nesterov=True
        )

        # Set the scheduler steps based on the trainer or steps if not using a Lightning trainer
        if hasattr(self, "trainer"):
            self.scheduler_steps = self.trainer.max_epochs
        else:
            assert hasattr(
                self, "steps"
            ), "You're not using a Lightning trainer. Please set the number of epochs in self.set_scheduler_steps(epochs)"
            self.scheduler_steps = self.steps

        # Define the learning rate scheduler (same as original nnUNet PolyLRScheduler)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.scheduler_steps, power=0.9)

        return [optimizer], [scheduler]

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.backbone,
            overlap=self.sw_overlap,
            mode=self.sw_mode,
        )
