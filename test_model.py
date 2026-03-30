from src.models.tcn_model import load_tcn_model
model, config = load_tcn_model()
print('Model loaded!')
print(f'Threshold : {config["threshold"]}')
print(f'Channels  : {config["n_channels"]}')
print(f'Timesteps : {config["n_timesteps"]}')
