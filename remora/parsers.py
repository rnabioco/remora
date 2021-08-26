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


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Helper function to prettier print subcommand help. This removes some
    extra lines of output when a final command parser is not selected.
    """

    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


#####################
# betta match_pairs #
#####################


def register_train_model(parser):
    subparser = parser.add_parser(
        "train_model",
        description="Train modified base model",
        help="Train modified base model",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "--dataset-path",
        default="toy_training_data.hdf5",
        help="Training dataset",
    )

    subparser.add_argument(
        "--results-path",
        default="results",
        help="Path to the output results file",
    )

    subparser.add_argument(
        "--output-file-type",
        default="txt",
        help="The file type of the output results",
    )
    subparser.add_argument(
        "--checkpoint-path",
        default="./models",
        help="Path to save the model and other auxiliary files",
    )

    subparser.add_argument(
        "--chunk_bases",
        default=[],
        type=tuple,
        help="sample smaller chunks from the reads according to bases before "
        "and after mod",
    )
    subparser.add_argument(
        "--num-chunks",
        default=1000,
        type=int,
        help="Number of samples per batch.",
    )

    subparser.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="Number of samples per batch.",
    )
    subparser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs."
    )
    subparser.add_argument(
        "--gpu-id",
        default=0,
        type=int,
        help="ID of GPU that is used for training.",
    )
    subparser.add_argument(
        "--workers",
        default=0,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    subparser.add_argument("--model", default="lstm", help="Model for training")
    subparser.add_argument(
        "--loss", default="CrossEntropy", help="Criterion for training"
    )
    subparser.add_argument(
        "--optimizer", default="adamw", help="Optimizer setting"
    )
    subparser.add_argument(
        "--lr", default=1e-5, type=float, help="Learning rate setting"
    )
    subparser.add_argument(
        "--lr-decay-step",
        default=10,
        type=int,
        help="Learning decay step setting",
    )
    subparser.add_argument(
        "--lr-decay-gamma",
        default=0.5,
        type=float,
        help="Learning decay gamma setting",
    )
    subparser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    subparser.add_argument(
        "--val-prop",
        default=0.1,
        type=float,
        help="Proportion of the dataset to be used as validation",
    )
    subparser.add_argument(
        "--mod-offset",
        default=20,
        type=int,
        help="Seed value",
    )
    subparser.add_argument(
        "--mod",
        default="a",
        type=str,
        help="The mod base being considered",
    )
    subparser.add_argument(
        "--motif",
        default="CAGT",
        type=str,
        help="The motif being considered",
    )
    subparser.add_argument(
        "--evenchunks",
        default="store_false",
        action="store_true",
        help="make all chunk sizes the same",
    )

    subparser.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="After how many epochs to save the model. Default 10.",
    )

    subparser.add_argument("--seed", default=1, type=int, help="Seed value")
    subparser.add_argument("--remark", default="", help="Any reamrk")

    subparser.set_defaults(func=run_train_model)


def run_train_model(args):
    from remora.train_model import train_model

    train_model(args)
