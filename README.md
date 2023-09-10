# PROJET FAIMG

This work is based on the following research papers of Ronny Hänsch.

Skipping the real world: Classification of PolSAR images without explicit feature extraction [[Ronny Hänsch et al., 2018]][0]

Fusion of Multispectral LiDAR, Hyperspectral, and RGB Data for Urban Land Cover Classification [[Ronny Hänsch et al., 2020]][1]

[0]: https://www.sciencedirect.com/science/article/abs/pii/S092427161730223X
[1]: https://ieeexplore.ieee.org/document/9007361

## Requirement

- Python 3.7 and up
- Rust 1.48 and up

## Installations

To install Python follow the steps : https://www.python.org/downloads/

To install Rust follow the steps : https://www.rust-lang.org/tools/install

To install python dependencies :

```
pip install -r requirements.txt
```

## How to compile ?

There are multiple ways, but an easy one is to make a virtual environment in python :

```
python -m venv .venv
source .venv/bin/activate
```

To build the module, you need to go to the `faimg_rs` folder and compile it using maturin, which is in the requirements.txt :

```
cd faimg_rs
maturin develop --release
```

It will make the module available in the current virtual environment.

Done!

#### Side notes

You can also use

```
maturin build --release
```

And it will give you the path to the compiled .whl wheel file in order to use the module elsewhere

For more informations visit : https://github.com/PyO3/pyo3 and https://github.com/PyO3/maturin

### If faimg_rs is too cumbersome

If faimg_rs leads to too much troubles, you can stop using it easily :

In `faimg.py`, comment the line

```python
from faimg_rs import get_best_split
```

And change

```python
        return self.findBestSplitRs(patches, x, y)
        # return self.findBestSplit(patches, x, y)
```

to

```python
        # return self.findBestSplitRs(patches, x, y)
        return self.findBestSplit(patches, x, y)
```

## Repository content

- `samples`: Sample images, made available by Humans in the Loop in this [dataset](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
- `faimg.py` : Random Forest of Hansch
- `faimgTest.py` : Tests for `faimg.py`
- `randomForest.py` : CART Tree and classic Random Forest
- `faimg.ipynb` : Demo
- `faimg_rs`: Folder with the source of the Python module written in Rust

## Demo

The demo is the jupiter notebook named `faimg.ipynb` in which the model is used to make prediction on small images.

The environment for the notebook needs to be the virtual environment created earlier.

## Test

To run the python tests, an interactive python file is available in `faimgTest.py`.

To run the rust tests, run `cargo test` in the `faimg_rs` directory.

## Limitations

The model cannot out of the box take a folder as input and learn every pictures in it.

Though the split is pretty fast, the feature calculations can be a little slow in its current state.
