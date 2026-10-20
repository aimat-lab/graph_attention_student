import json
import os
import sys
import rich
import rich_click
import rich_click as click
from rich.console import Console, ConsoleOptions, RenderResult
from rich.padding import Padding
from rich.text import Text
from rich.style import Style
from rich.panel import Panel
from rich.table import Table

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.util import dynamic_import

from pycomex.functional.experiment import run_experiment
from pycomex.cli import ExperimentCLI
from rich.columns import Columns
from rich.syntax import Syntax

from graph_attention_student.util import (
    get_version,
    TEMPLATE_PATH,
    TEMPLATE_ENV,
)
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.data import SmilesDataset
from graph_attention_student.torch.advanced import megan_prediction_report


class RichLogo:
    """
    A rich display which will show the ASlurmX logo in ASCII art when printed.
    """

    STYLE = Style(bold=True, color="white")

    def __rich_console__(self, console, options):
        text_path = os.path.join(TEMPLATE_PATH, "logo_text.txt")
        with open(text_path) as file:
            text_string: str = file.read()
            text = Text(text_string, style=self.STYLE)
            
        image_path = os.path.join(TEMPLATE_PATH, "logo_image.txt")
        with open(image_path) as file:
            image_string: str = file.read()
            # Replace \e with actual escape character and create Text from ANSI
            ansi_string = image_string.replace('\\e', '\033')
            image = Text.from_ansi(ansi_string)
            
        side_by_side = Columns([image, text], equal=True, padding=(0, 3))
        yield Padding(side_by_side, (1, 3, 0, 3))


class RichHelp:

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:

        yield "[white bold]MEGAN[/white bold] - Multi Explanation Graph Attention Network"
        yield ""
        yield (
            "MEGAN is an explainable graph neural network that provides both predictions and visual "
            "explanations for molecular property prediction tasks. This command line interface enables "
            "training custom MEGAN models on molecular datasets and generating predictions with "
            "comprehensive explanation reports."
        )
        yield ""
        # ~ Model Training
        yield "🚀 [magenta bold]Training Models[/magenta bold]"
        yield ""
        yield (
            "Train MEGAN models on CSV datasets containing SMILES strings and target values. "
            "The training process automatically handles molecular featurization, model optimization, "
            "and explanation consistency losses."
        )
        yield Padding(
            Syntax(
                ("megan train dataset.csv --prediction-mode regression --max-epochs 200"),
                lexer="bash",
                theme="monokai",
                line_numbers=False,
            ),
            (1, 3),
        )
        yield (
            "For classification tasks, specify the prediction mode and adjust final layer units:"
        )
        yield Padding(
            Syntax(
                ("megan train data.csv --prediction-mode classification --final-units 64,32,3"),
                lexer="bash",
                theme="monokai",
                line_numbers=False,
            ),
            (1, 3),
        )
        yield ("Use [cyan]megan train --help[/cyan] for detailed training options")
        yield ""

        # ~ Making Predictions
        yield "🔮 [magenta bold]Making Predictions[/magenta bold]"
        yield ""
        yield (
            "Generate predictions and visual explanations for molecular SMILES strings using "
            "trained models. Creates comprehensive PDF reports with molecular visualizations "
            "and explanation heatmaps."
        )
        yield Padding(
            Syntax(
                ('megan predict "CCO" --model-path model.ckpt --processing-path process.py'),
                lexer="bash",
                theme="monokai",
                line_numbers=False,
            ),
            (1, 3),
        )
        yield (
            "For batch predictions or custom visualization settings:"
        )
        yield Padding(
            Syntax(
                ('megan predict "c1ccccc1" --output-path benzene_report.pdf --width 1500 --height 1500'),
                lexer="bash",
                theme="monokai",
                line_numbers=False,
            ),
            (1, 3),
        )
        yield ("Use [cyan]megan predict --help[/cyan] for prediction options")
        yield ""

        # ~ Dataset Requirements
        yield "📊 [magenta bold]Dataset Format[/magenta bold]"
        yield ""
        yield (
            "Training datasets should be CSV files with SMILES strings and target values:"
        )
        yield Padding(
            Syntax(
                ("smiles,logP\nCCO,0.25\nc1ccccc1,2.13\nCCC,1.09"),
                lexer="csv",
                theme="monokai",
                line_numbers=False,
            ),
            (1, 3),
        )
        yield (
            "Supports [cyan]regression[/cyan] (continuous values), [cyan]binary classification[/cyan] (0/1), "
            "and [cyan]multi-class classification[/cyan] (multiple target columns)."
        )


class CLI(click.RichGroup):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cons = Console()
        
        self.add_command(self.train_command)
        self.add_command(self.predict_command)
        self.add_command(self.export_command)
        
    def format_help(self, ctx, formatter):
        
        rich.print(RichLogo())
        rich.print(Padding(RichHelp(), (1, 1)))
        
        self.format_usage(ctx, formatter)
        self.format_options(ctx, formatter)
        #self.format_commands(ctx, formatter)
        self.format_epilog(ctx, formatter)
        
    # === COMMANDS ===
        
    @click.command(
        'train',
        short_help='Train a MEGAN model based on a CSV of SMILES strings and target values.'
    )
    @click.argument('csv_path', type=click.Path(exists=True, readable=True))
    @click.option('--smiles-column', default='smiles',
                  help='Column name containing SMILES molecular representations (default: smiles)')
    @click.option('--target-columns', default='value',
                  help='Comma-separated target column names for prediction. For regression: single column. For classification: multiple columns for multi-class (default: value)')
    @click.option('--max-epochs', default=150,
                  help='Maximum number of training epochs. Training will stop early if convergence is reached (default: 150)')
    @click.option('--batch-size', default=64,
                  help='Mini-batch size for training. Larger values need more GPU memory but may improve stability (default: 64)')
    @click.option('--learning-rate', default=1e-4,
                  help='Learning rate for Adam optimizer. Lower values = slower but more stable training (default: 0.0001)')
    @click.option('--units', default='64,64,64',
                  help='Comma-separated hidden units for graph encoder layers. More layers = deeper network. Example: "32,64,32" (default: 64,64,64)')
    @click.option('--final-units', default='64,32,1',
                  help='Comma-separated units for final MLP layers. Last value must match number of targets: 1 for regression, num_classes for classification (default: 64,32,1)')
    @click.option('--prediction-mode', default='regression', type=click.Choice(['regression', 'bce', 'classification']),
                  help='Task type: "regression" for continuous values, "bce" for binary classification, "classification" for multi-class (default: regression)')
    @click.option('--importance-factor', default=1.0,
                  help='Weight for explanation consistency loss. Higher values = more emphasis on explanations. Set to 0 to disable explanations (default: 1.0)')
    @click.option('--sparsity-factor', default=0.5,
                  help='Weight for explanation sparsity loss. Higher values = sparser explanations (fewer highlighted atoms) (default: 0.5)')
    @click.option('--importance-offset', default=1.0,
                  help='Offset for importance scores. Higher values = sparser explanations. Typical range: 0.5-2.0 (default: 1.0)')
    @click.option('--output-path', default=None,
                  help='Output path for trained model checkpoint. If not specified, saves as "model.ckpt" in current directory')
    @click.option('--num-workers', default=4,
                  help='Number of parallel workers for data loading. Increase for faster data loading, decrease if running out of memory (default: 4)')
    @click.option('--accelerator', default='auto',
                  help='PyTorch Lightning accelerator: "auto" (automatic detection), "cpu", "gpu", "tpu" (default: auto)')
    @click.option('--devices', default='auto',
                  help='Number/IDs of devices to use: "auto", 1, "0,1", etc. Use "auto" for automatic selection (default: auto)')
    @click.pass_obj
    def train_command(self, csv_path, smiles_column, target_columns, max_epochs, batch_size,
                     learning_rate, units, final_units, prediction_mode, importance_factor,
                     sparsity_factor, importance_offset, output_path, num_workers, accelerator, devices):
        """
        Train a MEGAN (Multi-Explanation Graph Attention Network) model on molecular data.
        
        \b
        This command trains an explainable graph neural network on molecular datasets from CSV files.
        The model learns to predict molecular properties while simultaneously providing explanations
        showing which atoms and bonds are most important for each prediction.

        \b
        The training process will:
        1. Load and process SMILES strings from the CSV file
        2. Convert molecules to graph representations with node/edge features
        3. Train the MEGAN model with explanation consistency losses
        4. Save the trained model and processing module for later use

        \b
        Required CSV format: Must contain a column with SMILES strings and target value columns.
        Example CSV structure:
          smiles,logP
          CCO,0.25
          c1ccccc1,2.13

        \b
        Output files:
        - model.ckpt: Trained model checkpoint (can be loaded with Megan.load())
        - process.py: Processing module (contains MoleculeProcessing instance)

        \b
        Examples:
          Basic training:
            megan train data.csv
          Classification with custom settings:
            megan train data.csv --prediction-mode classification --final-units 64,32,3 --max-epochs 200
          High-performance training:
            megan train data.csv --batch-size 128 --num-workers 8 --devices 2
        
        \b
        CSV_PATH: Path to CSV file containing SMILES strings and target values
        
        """

        console = Console()

        # Parse comma-separated values
        target_columns_list = [col.strip() for col in target_columns.split(',')]
        units_list = [int(u.strip()) for u in units.split(',')]
        final_units_list = [int(u.strip()) for u in final_units.split(',')]

        # Set default output path
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "model.ckpt")

        # Create configuration table
        config_table = Table(show_header=False, box=None, padding=(0, 1))
        config_table.add_column("Parameter", style="bold cyan", width=20)
        config_table.add_column("Value", style="white")

        config_table.add_row("Dataset", csv_path)
        config_table.add_row("SMILES Column", smiles_column)
        config_table.add_row("Target Columns", str(target_columns_list))
        config_table.add_row("Max Epochs", str(max_epochs))
        config_table.add_row("Batch Size", str(batch_size))
        config_table.add_row("Learning Rate", str(learning_rate))
        config_table.add_row("Prediction Mode", prediction_mode)
        config_table.add_row("Output Path", output_path)

        panel = Panel(
            config_table,
            title="[bold white]MEGAN Model Training Configuration[/bold white]",
            border_style="cyan"
        )

        console.print()
        console.print(panel)
        console.print()

        try:
            # Initialize molecule processing
            console.print("🔧 Initializing molecule processing...")
            processing = MoleculeProcessing()

            # Create dataset
            console.print(f"📚 Loading dataset from {csv_path}...")
            dataset = SmilesDataset(
                dataset=csv_path,
                smiles_column=smiles_column,
                target_columns=target_columns_list,
                processing=processing,
                reservoir_sampling=True,
            )

            # Create data loader
            console.print("Setting up data loader...")
            loader_train = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True,
                num_workers=num_workers,
                prefetch_factor=2,
            )

            # Initialize model
            console.print("🤖 Initializing MEGAN model...")

            # Determine importance mode based on prediction mode
            importance_mode = 'regression' if prediction_mode == 'regression' else 'classification'

            model = Megan(
                node_dim=processing.get_num_node_attributes(),
                edge_dim=processing.get_num_edge_attributes(),
                units=units_list,
                final_units=final_units_list,
                prediction_mode=prediction_mode,
                learning_rate=learning_rate,
                importance_mode=importance_mode,
                importance_factor=importance_factor,
                sparsity_factor=sparsity_factor,
                importance_offset=importance_offset,
            )

            console.print(f"Model initialized with {processing.get_num_node_attributes()} node features and {processing.get_num_edge_attributes()} edge features")

            # Initialize trainer
            console.print("Setting up PyTorch Lightning trainer...")
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=False,
            )

            # Start training
            console.print("\n🚀 [bold green]Starting training...[/bold green]\n")

            trainer.fit(
                model,
                train_dataloaders=loader_train,
            )

            # Put model in evaluation mode
            model.eval()

            # Save the model
            console.print(f"\n💾 Saving trained model to {output_path}...")
            model.save(output_path)

            # Save the processing instance as process.py
            processing_path = os.path.join(os.path.dirname(output_path), "process.py")
            console.print(f"💾 Saving processing instance to {processing_path}...")

            processing_module_content = create_processing_module(processing)
            with open(processing_path, 'w') as f:
                f.write(processing_module_content)

            console.print("\n✅ [bold green]Training completed successfully![/bold green]")
            console.print(f"Model saved to: [cyan]{output_path}[/cyan]")
            console.print(f"Processing saved to: [cyan]{processing_path}[/cyan]")

        except Exception as e:
            console.print(f"\n❌ [bold red]Training failed:[/bold red] {str(e)}")
            sys.exit(1)

    @click.command(
        'export',
        short_help='Export a trained MEGAN model to ONNX for use outside Python.'
    )
    @click.option('--model-path', default='model.ckpt',
                  help='Path to trained MEGAN model checkpoint file (.ckpt) (default: model.ckpt)')
    @click.option('--processing-path', default='process.py',
                  help='Path to processing module (.py file). Contains the MoleculeProcessing instance used during training (default: process.py)')
    @click.option('--output-path', default='onnx_export',
                  help='Directory to write model.onnx, manifest.json, fixtures.json and featurization.json into (default: onnx_export)')
    @click.option('--smiles', default='CCO,c1ccccc1O,CC(=O)Oc1ccccc1C(=O)O,CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
                  help='Comma-separated SMILES used to trace the model and to generate the verification fixtures. Use several differently-sized molecules; molecules without bonds are not supported.')
    @click.option('--opset', default=18,
                  help='ONNX opset version. Must be at least 18, since the segment max needs ScatterND with reduction="max" (default: 18)')
    @click.pass_obj
    def export_command(self, model_path, processing_path, output_path, smiles, opset):
        """
        Export a trained MEGAN model to ONNX.

        \b
        Writes a directory containing:
          model.onnx          the model - predictions, node/edge importances,
                              graph embeddings and the variance head
          manifest.json       the data contract: hashes, tensor signature, runtime
                              versions, architecture and behavioural notes
          featurization.json  how to turn a molecule into the input tensors
          fixtures.json       example inputs with their expected outputs, so an
                              independent implementation can verify itself

        \b
        The graph takes tensors, not SMILES - a consumer builds those itself using
        featurization.json. It runs under ONNX Runtime in Python, Node and the browser.

        \b
        Required files (generated by 'megan train'):
          model.ckpt   the trained model checkpoint
          process.py   the processing module used during training
        """
        console = Console()
        console.print(Panel(
            '[bold]MEGAN ONNX Export[/bold]',
            border_style='blue',
        ))

        try:
            if not os.path.exists(processing_path):
                raise FileNotFoundError(f'Processing module not found: {processing_path}')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model file not found: {model_path}')

            console.print('🔧 Loading processing module...')
            processing = dynamic_import(processing_path).processing

            console.print('🤖 Loading trained model...')
            model = Megan.load(model_path)
            model.eval()

            smiles_list = [value.strip() for value in smiles.split(',') if value.strip()]
            console.print(f'📊 Processing {len(smiles_list)} example molecules...')
            graphs = [processing.process(value) for value in smiles_list]

            console.print('📦 Exporting to ONNX...')
            from graph_attention_student.torch.export import export_megan_onnx
            paths = export_megan_onnx(
                model, output_path, graphs, processing=processing, opset=opset)

            table = Table(show_header=True, header_style='bold')
            table.add_column('Artifact')
            table.add_column('Path')
            table.add_column('Size', justify='right')
            for role, path in paths.items():
                size = os.path.getsize(path) / 1024
                table.add_row(role, path, f'{size:.0f} KB')
            console.print(table)

            manifest_path = paths['manifest']
            with open(manifest_path) as file:
                manifest = json.load(file)
            if manifest['featurization'] is None and manifest['featurizationOmittedBecause']:
                console.print(
                    f'[yellow]No featurization.json was written:[/yellow] '
                    f'{manifest["featurizationOmittedBecause"]}'
                )
                console.print(
                    '[yellow]The model itself exported fine - only the JavaScript '
                    'featurization contract was withheld.[/yellow]'
                )

            console.print('[green]Export complete.[/green]')

        except Exception as exc:
            console.print(f'[red]Export failed:[/red] {exc}')
            raise click.Abort()

    @click.command(
        'predict',
        short_help='Generate predictions and explanations for a SMILES string using a trained MEGAN model.'
    )
    @click.argument('smiles', type=str)
    @click.option('--model-path', default='model.ckpt',
                  help='Path to trained MEGAN model checkpoint file (.ckpt). Must be compatible with the processing module (default: model.ckpt)')
    @click.option('--processing-path', default='process.py',
                  help='Path to processing module (.py file). Contains the MoleculeProcessing instance used during training (default: process.py)')
    @click.option('--output-path', default=None,
                  help='Output path for prediction report PDF. Contains molecular visualization with explanation heatmaps (default: prediction_report.pdf)')
    @click.option('--width', default=1000,
                  help='Width of molecular visualization in pixels. Higher values = more detailed images (default: 1000)')
    @click.option('--height', default=1000,
                  help='Height of molecular visualization in pixels. Higher values = more detailed images (default: 1000)')
    @click.pass_obj
    def predict_command(
        self, smiles, model_path, processing_path, output_path, width, height
    ):
        """
        Generate predictions and visual explanations for molecular SMILES strings.

        \b
        This command uses a trained MEGAN model to predict molecular properties and generate
        comprehensive explanation reports. The model provides both numerical predictions and
        visual explanations showing which atoms and bonds contribute most to the prediction.

        \b
        The prediction process:
        1. Loads the trained model and processing module
        2. Converts the SMILES string to a molecular graph representation
        3. Generates a property prediction using the trained model
        4. Creates explanation heatmaps showing atom/bond importance
        5. Produces a detailed PDF report with visualizations

        \b
        Required files (generated by 'megan train'):
        - Model checkpoint (.ckpt): Contains the trained neural network weights
        - Processing module (.py): Contains the molecular featurization logic

        \b
        Examples:
          Basic prediction:
            megan predict "CCO"
          Using custom files:
            megan predict "c1ccccc1" --model-path my_model.ckpt --processing-path my_process.py
          
        \b
        SMILES: SMILES string representation of the molecule to analyze (e.g., "CCO", "c1ccccc1")

        """
        
        console = Console()

        # Set default output path
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "prediction_report.pdf")

        # Create configuration table
        config_table = Table(show_header=False, box=None, padding=(0, 1))
        config_table.add_column("Parameter", style="bold cyan", width=20)
        config_table.add_column("Value", style="white")

        config_table.add_row("SMILES", smiles)
        config_table.add_row("Model Path", model_path)
        config_table.add_row("Processing Path", processing_path)
        config_table.add_row("Output Path", output_path)
        config_table.add_row("Image Size", f"{width}x{height}")

        panel = Panel(
            config_table,
            title="[bold white]MEGAN Prediction Configuration[/bold white]",
            border_style="cyan"
        )

        console.print()
        console.print(panel)
        console.print()

        try:
            # Load processing module
            console.print("🔧 Loading processing module...")
            if not os.path.exists(processing_path):
                raise FileNotFoundError(f"Processing module not found: {processing_path}")

            module = dynamic_import(processing_path)
            processing = module.processing
            console.print(f"Processing loaded with {processing.get_num_node_attributes()} node features and {processing.get_num_edge_attributes()} edge features")

            # Load trained model
            console.print("🤖 Loading trained model...")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = Megan.load(model_path)
            model.eval()
            console.print("Model loaded successfully")

            # Process SMILES into graph
            console.print(f"📊 Processing SMILES: {smiles}")
            try:
                graph = processing.process(smiles)
            except Exception as e:
                raise ValueError(f"Failed to process SMILES '{smiles}': {str(e)}")

            # Generate prediction
            console.print("🔮 Generating prediction...")
            results = model.forward_graph(graph)
            predicted_value = results['graph_output'].item()

            console.print(f"Predicted value: [bold green]{predicted_value:.4f}[/bold green]")

            # Generate prediction report with visualization
            console.print(f"📄 Generating prediction report...")
            megan_prediction_report(
                value=smiles,
                model=model,
                processing=processing,
                output_path=output_path,
                vis_width=width,
                vis_height=height,
            )

            console.print("\n✅ [bold green]Prediction completed successfully![/bold green]")
            console.print(f"Predicted value: [bold green]{predicted_value:.4f}[/bold green]")
            console.print(f"Report saved to: [cyan]{output_path}[/cyan]")

        except Exception as e:
            console.print(f"\n❌ [bold red]Prediction failed:[/bold red] {str(e)}")
            sys.exit(1)


@click.group(cls=CLI, invoke_without_command=True)
@click.option("-v", "--version", is_flag=True, help="Show the package version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Console script for pycomex."""

    ctx.obj = ctx.command

    if version:
        version = get_version()
        click.secho(version)
        sys.exit(0)


if __name__ == "__main__":
    cli()  # pragma: no cover
