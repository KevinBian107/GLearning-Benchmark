"""
Main training entry point for all graph learning models.
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_current_conda_env():
    """Get the name of the current conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    return conda_env


def check_conda_environment(model):
    """
    Check if the correct conda environment is active for the given model.

    Required environments:
    - mpnn, ibtt: glearning_180a
    - ggps: graphgps
    - agtt: autograph
    """
    env_requirements = {
        'mpnn': 'glearning_180a',
        'ibtt': 'glearning_180a',
        'ggps': 'graphgps',
        'agtt': 'autograph',
    }

    required_env = env_requirements.get(model)
    current_env = get_current_conda_env()

    if current_env is None:
        print("WARNING: No conda environment detected!")
        print(f"   This model ({model.upper()}) requires conda environment: {required_env}")
        print(f"   Please activate it with: conda activate {required_env}")
        print()
        return False

    if current_env != required_env:
        print("=" * 80)
        print("ERROR: Wrong conda environment!")
        print("=" * 80)
        print(f"Current environment: {current_env}")
        print(f"Required environment: {required_env}")
        print()
        print(f"To fix this, run:")
        print(f"  conda activate {required_env}")
        print(f"  python train.py --model {model} {'--config ' + sys.argv[sys.argv.index('--config') + 1] if '--config' in sys.argv else ''}")
        print("=" * 80)
        sys.exit(1)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train graph learning models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Index-Based Tokenization Transformer
  python train.py --model ibtt --config configs/ibtt_graph_token.yaml

  # Train Message Passing Neural Network
  python train.py --model mpnn --config configs/mpnn_graph_token.yaml

  # Train GraphGPS
  python train.py --model ggps --config configs/gps_graph_token.yaml

  # Train AutoGraph Trail Tokenization Transformer
  python train.py --model agtt --config configs/agtt_graph_token.yaml
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['ibtt', 'mpnn', 'ggps', 'agtt'],
        help='Model to train (ibtt: Index-Based Tokenization Transformer, '
             'mpnn: Message Passing Neural Network, '
             'ggps: GraphGPS, '
             'agtt: AutoGraph Trail Tokenization Transformer)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (default: configs/<model>_graph_token.yaml)'
    )

    args = parser.parse_args()

    # Check conda environment before proceeding
    check_conda_environment(args.model)

    # Use default config if not specified
    if args.config is None:
        default_configs = {
            'ibtt': 'configs/ibtt_graph_token.yaml',
            'mpnn': 'configs/mpnn_graph_token.yaml',
            'ggps': 'configs/gps_graph_token.yaml',
            'agtt': 'configs/agtt_graph_token.yaml',
        }
        args.config = default_configs[args.model]
        print(f"Using default config: {args.config}")
        print()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config and run appropriate trainer
    print("=" * 80)
    print(f"Training Model: {args.model.upper()}")
    print(f"Config: {args.config}")
    print("=" * 80)
    print()

    # Lazy import to avoid loading all dependencies at once
    if args.model == 'ibtt':
        from trainer import train_ibtt
        config = train_ibtt.load_config(args.config)
        print(f"Task: {config['dataset']['task']}")
        print(f"Train Algorithms: {config['dataset']['train_algorithms']}")
        print(f"Test Algorithm (OOD): {config['dataset']['test_algorithm']}")
        print()
        train_ibtt.main(config)

    elif args.model == 'mpnn':
        from trainer import train_mpnn
        config = train_mpnn.load_config(args.config)
        task = config['dataset']['task']
        print(f"Task: {task}")
        if 'train_algorithms' in config['dataset']:
            print(f"Train Algorithms: {config['dataset']['train_algorithms']}")
            print(f"Test Algorithm (OOD): {config['dataset']['test_algorithm']}")
        print()
        train_mpnn.main(config)

    elif args.model == 'ggps':
        from trainer import train_ggps
        config = train_ggps.load_config(args.config)
        task = config['data']['task']
        print(f"Task: {task}")
        if 'train_algorithms' in config['data']:
            print(f"Train Algorithms: {config['data']['train_algorithms']}")
            print(f"Test Algorithm (OOD): {config['data']['test_algorithm']}")
        print()
        train_ggps.main(config)

    elif args.model == 'agtt':
        from trainer import train_agtt
        config = train_agtt.load_config(args.config)
        task = config['dataset']['task']
        print(f"Task: {task}")
        if 'train_algorithms' in config['dataset']:
            print(f"Train Algorithms: {config['dataset']['train_algorithms']}")
            print(f"Test Algorithm (OOD): {config['dataset']['test_algorithm']}")
        print()
        train_agtt.main(config)


if __name__ == "__main__":
    main()
