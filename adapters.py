
# from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
from typing import Any, Dict, Type

# import io
import wandb
from utils import wandb_htmltable
# import base64
# from PIL import Image


# def on_save_checkpoint(checkpoint):
#     # pop the backbone here using custom logic
#     param_grad_dic = {
#         k: v.requires_grad for (k, v) in model.named_parameters()
#     }
#     state_dict = model_no_ddp.state_dict()
#     for k in list(state_dict.keys()):
#         if k in param_grad_dic.keys() and not param_grad_dic[k]:
#             # delete parameters that do not require gradient
#             del state_dict[k]
#     # del checkpoint['state_dict'][backbone_keys]


class SavingCallback(pl.Callback):

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        # asr_columns = ["Truth", "Transcription"]
        # pl_module.asr_table = pl_module.logger.log_table(key="ASR", columns=asr_columns)
        # ast_columns = ["Truth", "Translation"]
        # pl_module.ast_table = pl_module.logger.log_table(key="AST", columns=ast_columns)
        # langid_columns = ["Truth", "Lang Prediction"]
        # pl_module.lid_table = pl_module.logger.log_table(key="Lang ID", columns=langid_columns)

        # asr_columns = ["Truth", "Transcription"]
        # pl_module.asr_table = wandb.Table(columns=asr_columns)
        # # key="ASR", 
        # ast_columns = ["Truth", "Translation"]
        # pl_module.ast_table = wandb.Table(columns=ast_columns)
        # # key="AST", 
        # langid_columns = ["Truth", "Lang Prediction"]
        # pl_module.lid_table = wandb.Table(columns=langid_columns)




    def on_train_end(self, trainer, pl_module):
        print("Training is ending")



    # def on_validation_end(self, trainer, pl_module):
        # Your code here
        # print(pl_module.lid_table)
        # print(trainer.global_step)
        # lid_data = list(zip(pl_module.lang_truth, pl_module.lang_selected, pl_module.lang_id_acc))
        # lid_columns = pl_module.langid_columns
        # lid_html = wandb_htmltable(lid_data, lid_columns)
        # wandb.log({"Lang ID": lid_html}, step=trainer.global_step, commit=True)
        # print("Validation ended!")


    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        r"""Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
            trainer: the current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: the current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.

        """

        param_grad_dic = {
            k: v.requires_grad for (k, v) in pl_module.named_parameters()
        }
        # print(param_grad_dic)
        state_dict = pl_module.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        checkpoint['state_dict'] = state_dict




    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        r"""Called when loading a model checkpoint, use to reload state.

        Args:
            trainer: the current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: the current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: the full checkpoint dictionary that got loaded by the Trainer.

        """
        print("Loading checkpoint")
        for key in self.state_dict().keys():
            if key.startswith("clip_model"):
                checkpoint["state_dict"][key] = self.state_dict()[key]


    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     """
    #     CLIP is pretrained, so we already have access to its
    #     checkpoint and load it separately with `set_clip`.
    #     However, the PyTorch Lightning Trainer is strict about checkpoint loading (not
    #     configurable), so it expects the loaded state_dict to match exactly the keys in
    #     the model. See https://github.com/Lightning-AI/lightning/issues/13246
    #     So, when loading the checkpoint, before loading it, we add all clip keys
    #     to it, so that they match
    #     """
    #     for key in self.state_dict().keys():
    #         if key.startswith("clip_model"):
    #             checkpoint["state_dict"][key] = self.state_dict()[key]

    