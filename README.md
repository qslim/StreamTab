# StreamTab: Training-Free Temporal Link Prediction with Tabular Foundation Models

Code for the paper "StreamTab: Training-Free Temporal Link Prediction with Tabular Foundation Models".

## Environment

- Ubuntu 24.04
- Python 3.11
- PyTorch 2.3.1
- TabPFN 2.0.9
- NumPy 1.26.4
- pandas 2.3.2
- scikit-learn 1.6.1

```bash
pip install -r requirements.txt
```

## Pretrained Model

StreamTab uses the TabPFN-v2 classifier without updating its parameters. TabPFN downloads the weights on first use. To use a local copy, download `tabpfn-v2-classifier.ckpt` from [Prior-Labs/TabPFN-v2-clf](https://huggingface.co/Prior-Labs/TabPFN-v2-clf) and pass its path with `--model_path`.

## Datasets

Download the datasets from [Zenodo](https://zenodo.org/record/7213796) and preprocess them with [DyGLib](https://github.com/yule-BUAA/DyGLib). The code reads the resulting `ml_<dataset>.csv`. Any chronologically sorted CSV with the columns `u` (source), `i` (destination), and `ts` (timestamp) also works.

## Usage

```bash
python main.py --data processed_data/wikipedia/ml_wikipedia.csv --regime historical
```

`--regime` selects the negative sampling regime: `random`, `historical`, or `inductive`. The defaults are a context of `k=8192` rows, `g=1` negative per event, and a refresh every `R=4` batches.

The script uses the last 15% of the stream as the test period and prints the mean AP over its batches. Test negatives are drawn with the context sampler of the selected regime, so this value is not comparable with the benchmark results in the paper, which use the DyGLib negative samplers.

## Citation

Citation information will be updated upon publication.

## License

This code is released under the [MIT License](LICENSE). Built with PriorLabs-TabPFN. The TabPFN [package](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/LICENSE) and [pretrained weights](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt) are distributed under the Prior Labs License.
