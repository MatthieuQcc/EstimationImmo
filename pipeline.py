from TrainModel import TrainModel

if __name__ == "__main__": 
    model = TrainModel(csv_path="dataset_gouv.csv")
    model.train_model()
    model.test_model()