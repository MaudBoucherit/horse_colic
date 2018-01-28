# Horse Colic
This project's goal is to predict rather a horse needs surgery, given some of its symptoms.

Maud Boucherit, Jan 2018

When a horse suffer from colic, veterinarians have to report several symptoms of the horse. The list right now is long: 20 variables, from an estimate of the horse's pain to its rectal temperature, via the colour of its mucous membranes. But are all these features relevant to predict if the horse need surgery? This project is building and fitting a logistic regression on the data, with a random features elimination model. This model should help vets to focus on the most important symptoms.

You can find the final report of this project [here](doc/report.ipynb).


## Getting Started

First clone this repo on your local environment using:   
```
git clone https://github.com/MaudBoucherit/horse_colic.git
```

Then you can run all the project using:
```
make all
```

You can also empty the repo by running:
```
make clean
```

### Dependencies

This project is run using Python 3. The following packages must be installed:
- `argparse`
- `pandas`
- `numpy`
- `matplotlib.pyplot`
- `altair`
- `fancyimpute`
- `pickle`
- `sklearn`


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
