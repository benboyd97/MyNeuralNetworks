# Part 1

# Part 2

To train our Neural Network for part 2 run:
```
python part2_house_value_regression.py
```

This file contains the Regressor class that implements fit, predict, and score methods. It also performs hyperparameter tuning and saves the best model as 
```
part2_model.pickle
```
which can be loaded into a regressor instance.

The PyTorch model is stored in Regressor as the model attribute. This is defined in the separate NeuralNetwork class.