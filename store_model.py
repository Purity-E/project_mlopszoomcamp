import mlflow
import pickle

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("course_project")

# loading the model from mlflow
model_name = "FakeNews_Classifier"
stage = "Production"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

# storing the model locally
with open('models/model.pkl', "wb") as f_out:
        pickle.dump(model, f_out)