from TrainModel import TrainModel

if __name__ == "__main__": 
    model = TrainModel(csv_path="dataset_gouv.csv")
    model.train()
    model.test()