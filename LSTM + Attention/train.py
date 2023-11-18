from inputDataPrepration import encoder_input_data,decoder_input_data,decoder_target_data
from model import model
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


#configrations
epochs = 4
batch_size = 32
validation_split = 0.2
model_name = "seq2seqLSTM + Attention"

wandb.init(
    # set the wandb project where this run will be logged
    project="grammarly-reverse-engineered-base-model",

    # track hyperparameters and run metadata with wandb.config
    config={
        "data:":"C4 200M",
        "model": model_name,
        "files_taken":1,
        "n_rows":1000,
        "optimizer":"adam", 
        "loss":"categorical_crossentropy", 
        "metric": "accuracy",
        "latent_dim":100,
        "epoch": epochs,
        "batch_size": batch_size,
        "validation_split":validation_split
    }
)

config = wandb.config

model.fit([encoder_input_data,decoder_input_data],decoder_target_data,batch_size=batch_size,
          epochs=epochs,validation_split=validation_split,callbacks=[WandbMetricsLogger(log_freq=2),WandbModelCheckpoint("models")])

model.save("models/"+model_name)
wandb.finish()