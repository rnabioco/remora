"""Parsers module provides all implementations of command line interfaces.

Each command should implment a `register_` function and `run_` function. The
register function should take a parser and add the appropriate entries for the
command. The run function should accept the parser.parse_args() object and
execute the command.  run commands should contain minimal logic. If run
functions become too complex consider moving logic into an appropriate betta
module. Imports required for a particular command should be made inside of the
run commands to avoid loading all modules when a user simply wants the help
string.
"""

import argparse
import os
from pathlib import Path
from shutil import rmtree

from remora import constants
from remora import log, RemoraError

LOGGER = log.get_logger()


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Helper function to prettier print subcommand help. This removes some
    extra lines of output when a final command parser is not selected.
    """

    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


#####################################
# remora prepare_taiyaki_train_data #
#####################################


def register_prepare_taiyaki_train_data(parser):
    subparser = parser.add_parser(
        "prepare_train_data",
        description="Prepare Remora training dataset",
        help="Prepare Remora training dataset.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "mapped_signal_file",
        help="Taiyaki mapped signal file.",
    )
    subparser.add_argument(
        "--output-remora-training-file",
        default="remora_training_dataset.npz",
        help="Output Remora training dataset file. Default: %(default)s",
    )
    subparser.add_argument(
        "--motif",
        nargs=2,
        metavar=("MOTIF", "FOCUS_POSITION"),
        default=["N", "0"],
        help="Extract training chunks centered on a defined motif. Argument "
        "takes 2 values representing 1) sequence motif and 2) focus position "
        "within the motif. For example to restrict to CpG sites use "
        '"--motif CG 0". Default: Any context ("N 0")',
    )
    subparser.add_argument(
        "--chunk-context",
        default=constants.DEFAULT_CHUNK_CONTEXT,
        type=int,
        nargs=2,
        help="Number of context signal points to select around the central "
        "position. Default: %(default)s",
    )
    subparser.add_argument(
        "--min-samples-per-base",
        type=int,
        default=constants.DEFAULT_MIN_SAMPLES_PER_BASE,
        help="Minimum number of samples per base. This sets the size of the "
        "ragged arrays of chunk sequences. Default: %(default)s",
    )
    subparser.add_argument(
        "--kmer-context-bases",
        nargs=2,
        default=constants.DEFAULT_KMER_CONTEXT_BASES,
        type=int,
        help="Definition of k-mer (derived from the reference) passed into "
        "the model along with each signal position. Default: %(default)s",
    )
    subparser.add_argument(
        "--mod-bases",
        help="Single letter codes for modified bases to predict. Must "
        "provide either this or specify --base-pred.",
    )
    subparser.add_argument(
        "--base-pred",
        action="store_true",
        help="Train to predict bases (SNPs) and not mods.",
    )
    subparser.add_argument(
        "--max-chunks-per-read",
        type=int,
        default=10,
        help="Maxiumum number of chunks to extract from a single read. "
        "Default: %(default)s",
    )
    subparser.add_argument(
        "--log-filename",
        help="Log filename. Default: Don't output log file.",
    )

    subparser.set_defaults(func=run_prepare_train_data)


def run_prepare_train_data(args):
    import atexit

    from taiyaki.mapped_signal_files import MappedSignalReader

    from remora.util import Motif, validate_mod_bases, get_can_converter
    from remora.prepare_train_data import extract_chunk_dataset

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    LOGGER.info("Opening mapped signal files")
    input_msf = MappedSignalReader(args.mapped_signal_file)
    atexit.register(input_msf.close)
    alphabet_info = input_msf.get_alphabet_information()
    alphabet, collapse_alphabet = (
        alphabet_info.alphabet,
        alphabet_info.collapse_alphabet,
    )
    motif = Motif(*args.motif)
    if not args.base_pred and args.mod_bases is None:
        raise RemoraError(
            "Must specify either modified base or base prediction model "
            "type option."
        )
    elif args.base_pred and args.mod_bases is not None:
        raise RemoraError(
            "Must specify either modified base or base prediction model "
            "type option not both."
        )
    if args.base_pred:
        if alphabet != "ACGT":
            raise ValueError(
                "Base prediction is not compatible with modified base "
                "training data. It requires a canonical alphabet."
            )
        label_conv = get_can_converter(alphabet, collapse_alphabet)
    else:
        label_conv = validate_mod_bases(
            args.mod_bases, motif, alphabet, collapse_alphabet
        )
    extract_chunk_dataset(
        input_msf,
        args.output_remora_training_file,
        motif,
        args.chunk_context,
        args.min_samples_per_base,
        args.max_chunks_per_read,
        label_conv,
        args.base_pred,
        args.mod_bases,
        alphabet_info.mod_long_names,
        args.kmer_context_bases,
    )


######################
# remora train_model #
######################


def register_train_model(parser):
    subparser = parser.add_parser(
        "train_model",
        description="Train Remora model",
        help="Train Remora model",
        formatter_class=SubcommandHelpFormatter,
    )

    subparser.add_argument(
        "remora_dataset_path",
        help="Remora training dataset",
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--val-prop",
        default=constants.DEFAULT_VAL_PROP,
        type=float,
        help="Proportion of the dataset to be used as validation. "
        "Default: %(default)f",
    )
    data_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of samples per batch. Default: %(default)d",
    )
    data_grp.add_argument(
        "--chunk-context",
        type=int,
        nargs=2,
        help="Override chunk context from data prep. Number of context signal "
        "points to select around the central position.",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        type=int,
        help="Override kmer context bases from data prep. Definition of "
        "k-mer (derived from the reference) passed into the model along with "
        "each signal position.",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--output-path",
        default="remora_train_results",
        help="Path to the output files. Default: %(default)s",
    )
    out_grp.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="After how many epochs to save the model. Default %(default)d",
    )
    out_grp.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if existing.",
    )

    mdl_grp = subparser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--model", required=True, help="Model architecture file (required)"
    )
    mdl_grp.add_argument(
        "--size",
        type=int,
        default=constants.DEFAULT_NN_SIZE,
        help="Model layer size. Default: %(default)d",
    )

    train_grp = subparser.add_argument_group("Training Arguments")
    train_grp.add_argument(
        "--epochs",
        default=constants.DEFAULT_EPOCHS,
        type=int,
        help="Number of training epochs. Default: %(default)d",
    )
    train_grp.add_argument(
        "--optimizer",
        default=constants.OPTIMIZERS[0],
        choices=constants.OPTIMIZERS,
        help="Optimizer setting. Default: %(default)s",
    )
    train_grp.add_argument(
        "--lr",
        default=constants.DEFAULT_LR,
        type=float,
        help="Learning rate setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--lr-decay-step",
        default=constants.DEFAULT_DECAY_STEP,
        type=int,
        help="Learning decay step setting. Default: %(default)d",
    )
    train_grp.add_argument(
        "--lr-decay-gamma",
        default=constants.DEFAULT_DECAY_GAMMA,
        type=float,
        help="Learning decay gamma setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--weight-decay",
        default=constants.DEFAULT_WEIGHT_DECAY,
        type=float,
        help="Weight decay setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--seed", default=1, type=int, help="Seed value. Default: %(default)d"
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for training. Default: Use CPU.",
    )

    subparser.set_defaults(func=run_train_model)


def run_train_model(args):
    from remora.train_model import train_model

    out_path = Path(args.output_path)
    if args.overwrite:
        if out_path.is_dir():
            rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
    elif out_path.exists():
        raise RemoraError("Refusing to overwrite existing training directory.")
    out_path.mkdir(parents=True, exist_ok=True)
    log.init_logger(os.path.join(out_path, "log.txt"))
    train_model(
        args.seed,
        args.device,
        out_path,
        args.remora_dataset_path,
        args.chunk_context,
        args.kmer_context_bases,
        args.val_prop,
        args.batch_size,
        args.model,
        args.size,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.lr_decay_step,
        args.lr_decay_gamma,
        args.epochs,
        args.save_freq,
    )


#######################
# remora export_model #
#######################


def register_export_model(parser):
    subparser = parser.add_parser(
        "export_model",
        description="Export a model to ONNX format for inference.",
        help="Export a model to ONNX format for inference.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "checkpoint_path",
        help="Path to a pretrained model checkpoint.",
    )
    subparser.add_argument(
        "output_path",
        help="Path to save the onnx model file.",
    )
    subparser.add_argument(
        "--model-path",
        help="Path to a model architecture. Default: Use path from checkpoint.",
    )

    subparser.set_defaults(func=run_export_model)


def run_export_model(args):
    from remora.model_util import continue_from_checkpoint, export_model

    LOGGER.info("Loading model")
    ckpt, model = continue_from_checkpoint(
        args.checkpoint_path, args.model_path
    )
    LOGGER.info("Exporting model to ONNX format")
    export_model(ckpt, model, args.output_path)


################
# remora infer #
################


def register_infer(parser):
    subparser = parser.add_parser(
        "infer",
        description="Run a model for inference on a given dataset.",
        help="Use modified base model for inference.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "dataset_path",
        help="Taiyaki mapped signal file on which to perform inference.",
    )
    subparser.add_argument(
        "onnx_model",
        help="Path to a pretrained model in onnx format.",
    )
    subparser.add_argument(
        "--focus-offset",
        type=int,
        help="Offset into stored chunks to be inferred. Default: Call all "
        "matches to motif (retrieved from model)",
    )
    subparser.add_argument(
        "--output-path",
        default="remora_infer_results",
        help="Path to the output files. Default: %(default)s",
    )
    subparser.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU",
    )
    subparser.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="Number of input units per batch. Default: %(default)d",
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if existing.",
    )

    subparser.set_defaults(func=run_infer)


def run_infer(args):
    import atexit

    from taiyaki.mapped_signal_files import MappedSignalReader

    from remora.inference import infer

    out_path = Path(args.output_path)
    if args.overwrite:
        if out_path.is_dir():
            rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
    elif out_path.exists():
        raise RemoraError("Refusing to overwrite existing inference results.")
    out_path.mkdir(parents=True, exist_ok=True)
    log.init_logger(os.path.join(out_path, "log.txt"))

    LOGGER.info("Opening mapped signal files")
    input_msf = MappedSignalReader(args.dataset_path)
    atexit.register(input_msf.close)

    infer(
        input_msf,
        out_path,
        args.onnx_model,
        args.batch_size,
        args.device,
        args.focus_offset,
    )
