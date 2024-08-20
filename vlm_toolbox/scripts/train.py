import os
import argparse
import warnings
import torch
import json
import pandas as pd
from matplotlib import pyplot as plt

from config.enums import (
    CLIPBackbones,
    ImageDatasets,
    LossType,
    Metrics,
    ModelType,
    PrecisionDtypes,
    SamplingStrategy,
    SamplingType,
    Setups,
    Sources,
    Stages,
    Trainers,
)
from config.logging import LoggerFactory
from config.setup import Setup
from pipeline.pipeline import Pipeline
from util.memory import flush

def json_to_dict(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string: {e}")

def main(args):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    
    logger = LoggerFactory.create_logger("coop_finetuning_logger")

    setup = Setup(
        dataset_name=args.dataset_name,
        backbone_name=args.backbone_name,
        trainer_name=args.trainer_name,
        model_type=args.model_type,
        setup_type=args.setup_type,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        validation_size=args.validation_size,
        label_column_name=args.label_column_name,
        n_shots=args.n_shots,
        precision_dtype=args.precision_dtype,
        load_from_checkpoint=args.load_from_checkpoint,
        loss_type=args.loss_type,
        annotations_key_value_criteria=args.annotations_key_value_criteria,
        train_full_precision=args.train_full_precision,
        eval_full_precision=args.eval_full_precision,
        enable_tensor_float=args.enable_tensor_float,
        source=args.source,
        main_metric_name=args.main_metric_name,
        preprocess_batch_size=args.preprocess_batch_size,
        train_split=args.train_split,
        eval_split=args.eval_split,
        top_k=args.top_k,
        random_state=args.random_state,
        model_checkpoint_path=args.model_checkpoint_path,
        auto_batch_size=args.auto_batch_size,
        use_dataset_context_init=args.use_dataset_context_init,
        do_augmentation=args.do_augmentation,
        loss_kwargs=args.loss_kwargs,
        sampling_type=args.sampling_type if args.sampling_type else None,
        sampling_strategy=args.sampling_strategy if args.sampling_strategy else None,
        sampling_kwargs=args.sampling_kwargs
    )
    setup.get_relative_save_path()
    DEVICE = torch.device(args.device_type)

    pipeline = Pipeline(setup, device_type=args.device_type, logger=logger)
    pipeline.run(
        collate_all_m2_samples=args.collate_all_m2_samples,
        save_predictions=args.save_predictions,
        persist=True,
    )
    pipeline.tear_down()
    del pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model")

    parser.add_argument("--dataset_name", type=str, required=True, choices=ImageDatasets.get_values(), help="Name of the dataset")
    parser.add_argument("--backbone_name", type=str, required=True, choices=CLIPBackbones.get_values(), help="Name of the backbone model")
    parser.add_argument("--trainer_name", type=str, required=True, choices=Trainers.get_values(), help="Name of the trainer")
    
    parser.add_argument("--model_type", type=str, default=ModelType.FEW_SHOT, choices=ModelType.get_values(), help="Type of the model")
    parser.add_argument("--setup_type", type=str, default=Setups.FULL, choices=Setups.get_values(), help="Setup type")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Evaluation batch size")
    parser.add_argument("--validation_size", type=float, default=0.15, help="Validation size")
    parser.add_argument("--label_column_name", type=str, default=None, help="Label column name")
    parser.add_argument("--n_shots", type=int, default=16, help="Number of shots")
    parser.add_argument("--precision_dtype", type=str, default=PrecisionDtypes.FP16, choices=PrecisionDtypes.get_values(), help="Precision dtype")
    parser.add_argument("--load_from_checkpoint", type=bool, default=False, help="Load from checkpoint")
    parser.add_argument("--loss_type", type=str, default=LossType.CONTRASTIVE_LOSS, choices=LossType.get_values(), help="Loss type")
    parser.add_argument("--annotations_key_value_criteria", type=json_to_dict, default='{}', help="JSON string of annotations key value criteria")
    
    parser.add_argument("--train_full_precision", type=bool, default=False, help="Train full precision")
    parser.add_argument("--eval_full_precision", type=bool, default=False, help="Eval full precision")
    parser.add_argument("--enable_tensor_float", type=bool, default=None, help="Enable tensor float")
    parser.add_argument("--source", type=str, default=Sources.OPEN_AI, choices=Sources.get_values(), help="Source")
    parser.add_argument("--main_metric_name", type=str, default=Metrics.ACCURACY, choices=Metrics.get_values(), help="Main metric name")
    parser.add_argument("--preprocess_batch_size", type=int, default=None, help="Preprocess batch size")
    parser.add_argument("--train_split", type=str, default=Stages.TRAIN, choices=Stages.get_values(), help="Train split")
    parser.add_argument("--eval_split", type=str, default=Stages.EVAL, choices=Stages.get_values(), help="Eval split")
    parser.add_argument("--top_k", type=int, default=5, help="Top K accuracy")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("--model_checkpoint_path", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--auto_batch_size", type=bool, default=False, help="Auto batch size")
    parser.add_argument("--use_dataset_context_init", type=bool, default=False, help="Use dataset context init")
    parser.add_argument("--do_augmentation", type=bool, default=False, help="Do augmentation")
    parser.add_argument("--loss_kwargs", type=json_to_dict, default='{}', help="JSON string of loss kwargs")
    parser.add_argument("--sampling_type", type=str, default=None, choices=SamplingType.get_values(), help="Sampling type")
    parser.add_argument("--sampling_strategy", type=str, default=None, choices=SamplingStrategy.get_values(), help="Sampling strategy")
    parser.add_argument("--sampling_kwargs", type=json_to_dict, default='{}', help="JSON string of sampling kwargs")

    default_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument("--device_type", type=str, default=default_device_type, choices=["cpu", "cuda"], help="Device type (default is based on availability)")

    parser.add_argument("--collate_all_m2_samples", type=bool, default=False, help="Collate all m2 samples")
    parser.add_argument("--save_predictions", type=bool, default=False)

    args = parser.parse_args()

    main(args)
    flush()