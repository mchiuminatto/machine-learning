import pandas as pd

def open_dataset(file_name: str) -> pd.DataFrame:
    titanic = pd.read_csv(file_name)
    titanic_train = titanic.drop(columns=["Name", "Ticket", "Cabin"])
    titanic_train["Sex"].replace({"male":1, "female":0}, inplace=True)
    titanic_train["Embarked"].replace({"S":1, "C":2, "Q":3}, inplace=True)
    return titanic_train
    
    