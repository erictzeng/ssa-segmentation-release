import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np



@click.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--output_dir', '-o', default='plots')
@click.option('--filter', '-f')
def main(results_dir, output_dir, filter):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    if filter is not None:
        suffix = f'-{filter}-only.pdf'
    else:
        suffix = '.pdf'
    output_path = output_dir / (results_dir.absolute().name + suffix)
    results_path = results_dir / 'results.pkl'
    if not results_dir.is_dir() or not results_path.exists():
        raise FileNotFoundError
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    iterations = sorted(results)
    x = []
    ys = {}
    for iteration in iterations:
        if iteration == 1:
            continue
        x.append(iteration)
        eval_results = results[iteration]
        for key, item in eval_results.items():
            if filter is not None and not key.startswith(filter):
                continue
            if key not in ys:
                ys[key] = []
            if isinstance(item, np.ndarray):
                item = np.mean(item)
            ys[key].append(item)
    for name, pts in ys.items():
        plt.plot(x, pts, label=name)
    plt.legend()
    plt.savefig(str(output_path))

if __name__ == '__main__':
    main()
