# Titanic Survival Prediction — Decision Tree Classifier

I built this to get hands-on with a real ML pipeline from scratch — not a toy 
dataset, but 891 actual Titanic passengers with messy, incomplete data that 
needed cleaning before anything useful could happen.

## What the project does

Takes passenger information (class, sex, age, fare, port of boarding, family 
size) and predicts whether they survived. The model lands at around 80% 
accuracy on the test set, which is respectable for a Decision Tree without 
any hyperparameter tuning beyond limiting depth.

## The pipeline

1. Load and explore the raw data
2. Handle missing values — Age filled with median, Embarked with mode, 
   Cabin dropped entirely (687 missing out of 891 rows, not worth keeping)
3. Drop columns that don't help prediction — Name, Ticket, PassengerId
4. One-hot encode Sex and Embarked
5. Train/test split (75/25)
6. Train a Decision Tree Classifier using entropy as the split criterion
7. Evaluate with accuracy score, confusion matrix, and classification report
8. Visualize the full decision tree
9. Run prediction on a new passenger with probability scores

## What I learned

Cleaning the data took more thought than training the model. Deciding what 
to drop, what to fill, and how to fill it actually matters — bad inputs 
produce bad outputs no matter how good the algorithm is.

## Libraries

pandas, scikit-learn, matplotlib

## How to run

1. Clone this repository
2. Make sure titanic.csv is in the same folder as the notebook
3. Open titanic.ipynb in Jupyter Notebook
4. Run All cells top to bottom
