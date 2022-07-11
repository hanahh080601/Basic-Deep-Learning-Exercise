# Basic-Deep-Learning-Exercise
# Build a basic Neural Network using pure Numpy and Pandas.

## Tasks 
* Preprocessing data
  * Loading data
  * Transform data
  * Split data
* Building model
  * Initialize weights & biases
  * Activation function
  * Loss function
  * Feed forward
  * Backpropagation
  * Update weights
* Training
  * * Execute (train, validate, test).
* Evaluating
  * Accuracy: approx 80 - 83%


## Installation

Clone the repo from Github and pull the project.
```bash
git clone https://github.com/hanahh080601/Basic-Deep-Learning-Exercise.git
git checkout hanlhn/multi-layers-nn
git pull
cd hanlhn/hanlhn
poetry install
poetry config virtualenvs.in-project true
poetry update
```

# Project tree 
.  
├── hanlhn          
│     ├── .venv             
│     ├── poetry.lock    
│     ├── pyproject.toml   
│     ├── README.rst  
│     └── hanlhn   
│           ├── __pycache__  
│           ├── data         
│           │      ├── dataset 
│           │      │        ├── test_record.csv  
│           │      │        └── train_record.csv      
│           │      └── datapipeline.py           
│           ├── models      
│           │      ├── TwoLayersNN.py  
│           │      ├── FCL.py      
│           │      └── MultiLayersNN.py       
│           ├── tests     
│           │      ├── __init__.py    
│           │      └── test_hanlhn.py     
│           ├── __init__.py     
│           ├── config.py      
│           └── main.py    
├── .gitignore                    
└── README.md   

## Usage: 
```bash
cd hanlhn/hanlhn
python main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author
[Lê Hoàng Ngọc Hân - Đại học Bách Khoa - Đại học Đà Nẵng (DUT)](https://github.com/hanahh080601) 