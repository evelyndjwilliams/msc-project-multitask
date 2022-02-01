import torch
import torch.nn as nn
from train import ConvNet2D, TORGODataset
import matplotlib.pyplot as plt
import json
from env import AttrDict
import os
import numpy as np
import librosa
import librosa.display
import pathlib

def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predicted = model(input) # tensor with dims: 1= input, 2=classes predicted
        expected = target
    test_loss = loss(predicted.squeeze(0), expected.squeeze(0))
    print(f"loss = {test_loss}")
    return predicted, expected, test_loss

if __name__ == "__main__":

    # edit to read this in from config json / AttrDict
    config_file = r"../resources/hifigan-config.json"
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    converter_config_file = r"./config.json"
    with open(converter_config_file) as f:
        data = f.read()

    global c
    json_config = json.loads(data)
    c = AttrDict(json_config)

    # validation
    SOURCE_ANNOTATIONS_FILE = c.test_source_annotations_file
    SOURCE_AUDIO_DIR = c.test_source_audio_dir
    TARGET_ANNOTATIONS_FILE = c.test_target_annotations_file
    TARGET_AUDIO_DIR = c.test_target_audio_dir

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)
    MIN_MEL = c.min
    MAX_MEL = c.max
    BATCH_SIZE = c.batch_size
    MAX_INPUT_LENGTH = c.longest
    N_MEL_FEATURES = c.n_mels
    EPOCHS = c.epochs
    LEARNING_RATE = c.learning_rate
    DECAY_RATE = c.lr_decay
    MODEL_NAME = c.model_name

    global best_loss
    best_loss = 10000 # gets replaced by training loss during first epoch


    output_path = f"../predicted/TORGO/{MODEL_NAME}"
    plot_output_path = f"./plots/TORGO/{MODEL_NAME}"

    if not os.path.exists(output_path):
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(plot_output_path):
        pathlib.Path(plot_output_path).mkdir(parents=True, exist_ok=True)

    dataset = TORGODataset(SOURCE_ANNOTATIONS_FILE,
                                                SOURCE_AUDIO_DIR,
                                                TARGET_ANNOTATIONS_FILE,
                                                TARGET_AUDIO_DIR,
                                                min_mel = MIN_MEL,
                                                max_mel = MAX_MEL,
                                                which_device = device,
                                                parameter_dict = h,
                                                align=True,
                                                max_input_length=MAX_INPUT_LENGTH)

    cnn = ConvNet2D()
    state_dict = torch.load(f"../trained_models/{MODEL_NAME}.pth",  map_location=torch.device('cpu'))
    # cnn = ConvNet(input_length=MAX_INPUT_LENGTH, n_mels=N_MEL_FEATURES)
    cnn.load_state_dict(state_dict)

    for i in range(len(dataset)):
        model = cnn
        model.eval()
        inputs, targets = (dataset[i][0]), (dataset[i][-1])
        with torch.no_grad():
            predicted = model(inputs.cpu())
        p = predicted.squeeze(1).squeeze(0)
        p *= MAX_MEL - MIN_MEL
        p += MIN_MEL
        torch.save(p.cpu(), os.path.join(output_path, f"{i}.pt"))
        plt.imshow(p, cmap='magma',origin='lower', interpolation='nearest')
        plt.savefig(os.path.join(plot_output_path, f"{i}_predicted_dysarthric.png"))
        inp = inputs.squeeze(1).squeeze(0)
        inp *= MAX_MEL - MIN_MEL
        inp += MIN_MEL
        plt.imshow(inp.cpu(), cmap='magma',origin='lower', interpolation='nearest')
        plt.savefig(os.path.join(plot_output_path, f"{i}_original_healthy.png"))
        e = targets.squeeze(1).squeeze(0)#.transpose(0, 1)
        e *= MAX_MEL - MIN_MEL
        e += MIN_MEL
        plt.imshow(e.cpu(), cmap='magma',origin='lower', interpolation='nearest')
        plt.savefig(os.path.join(plot_output_path, f"{i}_expected_dysarthric.png"))
