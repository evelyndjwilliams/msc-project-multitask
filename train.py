import os
import shutil
import pandas as pd
import json
import gc
import matplotlib.pyplot as plt
import time
import numpy
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchaudio
from torchaudio import datasets
from torchaudio.transforms import MFCC
from librosa.util import normalize
from dtw import dtw, warp
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from env import AttrDict


def reorganise_data(): # reorganises Voice Conversion Challenge dataset (don't need to do this again)
    with open("labels.txt", "w+") as f:
        shutil.rmtree(TRAINING_DATA_DIR)
        os.mkdir(TRAINING_DATA_DIR)
        for fold in os.listdir(OLD_DATA_DIR):
            label = fold[0]
            for file in os.listdir(os.path.join(OLD_DATA_DIR, fold)):
                shutil.copyfile(os.path.join(OLD_DATA_DIR, fold, file), os.path.join(TRAINING_DATA_DIR, f"{label}{file}"))
                f.write(f"{label}{file}, {label}\n")

class ConvNet2D(nn.Module):
    """ Defines architecture and training procedure for model.
        Model is 10-layer fully convolutional CNN with speaker embeddings.
        Trained using multi-task paradigm with Electromagnetic Articulograph features as secondary output."""

    def __init__(self):
        """Model architecture"""
        super(ConvNet2D, self).__init__()
        # encoder layers (regular convolution) - each has a batchnorm layer
        self.conv_enc_1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_enc_1_bn = nn.BatchNorm2d(256)
        self.conv_enc_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_enc_2_bn = nn.BatchNorm2d(256)
        self.conv_enc_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_enc_3_bn = nn.BatchNorm2d(256)
        self.conv_enc_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_enc_4_bn = nn.BatchNorm2d(256)
        self.conv_enc_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_enc_5_bn = nn.BatchNorm2d(256)
        # decoder layers (transpose convolution) (upsamples to reassume original length) - each has a batchnorm layer
        self.conv_dec_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_dec_1_bn = nn.BatchNorm2d(256)
        self.conv_dec_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_dec_2_bn = nn.BatchNorm2d(256)
        self.conv_dec_3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_dec_3_bn = nn.BatchNorm2d(256)
        self.conv_dec_4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_dec_4_bn = nn.BatchNorm2d(256)
        self.conv_dec_5 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv_dec_5_bn = nn.BatchNorm2d(1)

        self.ema_dec_dense = nn.Linear(in_features = 82, out_features=10)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.dense1 = nn.Linear(in_features=82, out_features=80)
        self.embedding = nn.Embedding(15, 2)
        self.dense2 = nn.Linear(50, 10)

    def forward(self, m_s, speaker_ID):
        """Training procedure"""
        # speaker embedding - tile to length of mel-spectrogram and concatenate to each frame
        e = self.embedding(torch.LongTensor([speaker_ID]).to(device)).transpose(0,1)
        e = e.detach().cpu().numpy()
        e = torch.tensor(numpy.tile(e, (list(m_s.size())[-1]))).to(device) # do in numpy because torch.tile() and torch.repeat() aren't working on uni gpu
        x = torch.cat((m_s.squeeze(0).squeeze(0), e), axis=0)
        x = x.unsqueeze(0).unsqueeze(0).to(device)

        x = self.relu(self.conv_enc_1_bn(self.conv_enc_1(x)))
        x = self.relu(self.conv_enc_2_bn(self.conv_enc_2(x)))
        x = self.relu(self.conv_enc_3_bn(self.conv_enc_3(x)))
        x = self.relu(self.conv_enc_4_bn(self.conv_enc_4(x)))
        x = self.relu(self.conv_enc_5_bn(self.conv_enc_5(x)))
        x = self.relu(self.conv_dec_1_bn(self.conv_dec_1(x)))
        x = self.relu(self.conv_dec_2_bn(self.conv_dec_2(x)))
        x = self.relu(self.conv_dec_3_bn(self.conv_dec_3(x)))
        x = self.relu(self.conv_dec_4_bn(self.conv_dec_4(x)))
        x = self.relu(self.conv_dec_5_bn(self.conv_dec_5(x)))

        m_s = self.dense1(x.transpose(2,3)) # predict mel-spectrogram features from final convolutional layer
        m_s_predictions = m_s.transpose(2,3)
        ema = self.ema_dec_dense(x.transpose(2,3)) # predict ema features from final convolutional layer
        ema_predictions = ema.transpose(2,3)

        return m_s_predictions, ema_predictions

class TORGODataset(Dataset):

    def __init__(self,
                ANNOTATIONS_FILE,
                AUDIO_DIR,
                REF_ANNOTATIONS_FILE,
                REF_AUDIO_DIR,
                min_mel,
                max_mel,
                ID_dict,
                ema_dir,
                ema_list,
                ema_min,
                ema_max,
                parameter_dict=None,
                max_input_length=858,
                which_device = "cuda",
                align=True
                ):
        self.annotations = pd.read_csv(ANNOTATIONS_FILE, header=None)
        self.ref_annotations = pd.read_csv(REF_ANNOTATIONS_FILE, header=None)
        self.audio_dir = AUDIO_DIR
        self.ref_audio_dir = REF_AUDIO_DIR
        self.align = align if __name__=="__main__" else False
        self.min_mel = min_mel
        self.max_mel = max_mel
        self.max_input_length = max_input_length
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.longest = 0 # will be updated
        self.parameter_dict = parameter_dict
        self.ID_dict = ID_dict
        self.ema_dir = ema_dir
        self.ema_list = pd.read_csv(ema_list, header=None)
        self.ema_min = ema_min
        self.ema_max = ema_max

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """returns processed healthy and dysarthric mel-spectrograms, EMA feature tensors, audio file ID and signal length"""
        print(index)
        healthy_sample_path = self.get_audio_sample_path(index,
                                                        audio_dir=self.audio_dir,
                                                        annotations=self.annotations)
        healthy_label = self.get_audio_sample_label(index)
        dysarthric_sample_path = self.get_audio_sample_path(index,
                                                        audio_dir=self.ref_audio_dir,
                                                        annotations=self.ref_annotations)
        dysarthric_label = self.get_audio_sample_label(index)
        dysarthric, sr = self.get_mel_spectrogram(dysarthric_sample_path)
        healthy, _ = self.get_mel_spectrogram(healthy_sample_path)
        signal_length = list(dysarthric.size())[2]

        if self.align == True: # do DTW on mel spectrogram frames
            healthy_mfcc = self.get_mfcc(healthy_sample_path, sr).squeeze(0).transpose(0,1)
            dysarthric_mfcc = self.get_mfcc(dysarthric_sample_path, sr).squeeze(0).transpose(0,1)
            dysarthric = dysarthric.squeeze(0).transpose(0,1)
            healthy = healthy.squeeze(0).transpose(0,1)

        """To warp dysarthric sample to healthy sample"""
        #     alignment = dtw(dysarthric.cpu(), healthy.cpu(), step_pattern="symmetric1")
        #     wq = warp(alignment)
        #     dysarthric_mel_spec = dysarthric[wq]
        # dysarthric_mel_spec = (dysarthric_mel_spec - self.min_mel) / self.max_mel # min-max normalisation
        # healthy = (healthy - self.min_mel) / self.max_mel
        # padded_signal = self.pad_signal(dysarthric_mel_spec)
        # healthy_mel_spec = self.pad_signal(healthy)
        # padded_signal = dysarthric_mel_spec#.transpose(0,1)
        # healthy_mel_spec = healthy[:-1]#.transpose(0,1)

        """To warp healthy sample to dysarthric sample"""
            alignment = dtw(healthy_mfcc.cpu(),dysarthric_mfcc.cpu(),  step_pattern="symmetric1")
            wq = warp(alignment)
            wq = numpy.where(wq== list(healthy.size())[0],  list(healthy.size())[0]-1, wq)
            healthy = healthy[wq]
            healthy = healthy.transpose(0,1)
            dysarthric = dysarthric.transpose(0,1)

        dysarthric_mel_spec = self.min_max_norm(dysarthric)
        healthy_mel_spec = self.min_max_norm(healthy)
        # plt.imshow(dysarthric_mel_spec, cmap='magma',origin='lower', interpolation='nearest')
        # plt.show()
        # plt.imshow(healthy_mel_spec, cmap='magma',origin='lower', interpolation='nearest')
        # plt.show()
        ema_features = self.get_ema_features(self.get_ema_sample_path(index, self.ema_dir)) # read EMA features in from file
        ema_len = len(ema_features[1,:])

        """Normalise length of EMA sequence and mel-spectrogram (EMA file lengths were inconsistent with audio length)"""
        if len(wq) <= ema_len:
            ema_index = len(wq)
            ema_features = ema_features[:, :ema_index]
        elif len(wq) > ema_len:
            ema_index = ema_len
            print(list(dysarthric_mel_spec.size()))
            dysarthric_mel_spec = dysarthric_mel_spec[:, :ema_index]
            healthy_mel_spec = healthy_mel_spec[:, :ema_index]

        dysarthric_mel_spec = dysarthric_mel_spec.unsqueeze(0)
        healthy_mel_spec = healthy_mel_spec.unsqueeze(0)
        target_ID = self.get_speaker_ID(index, annotations=self.ref_annotations)

        return healthy_mel_spec.to(self.device), target_ID, signal_length, ema_features, dysarthric_mel_spec.to(self.device)

    def min_max_norm(self, m_s):
        """Rescale features to [0, 1] range"""
        return (m_s- self.min_mel) /( self.max_mel- self.min_mel)

    def get_mel(self, signal):
        """Extract mel-spectrogram from audio signal"""
        return mel_spectrogram(signal.cpu(),
                                self.parameter_dict.n_fft,
                                self.parameter_dict.num_mels,
                                self.parameter_dict.sampling_rate,
                                self.parameter_dict.hop_size,
                                self.parameter_dict.win_size,
                                self.parameter_dict.fmin,
                                self.parameter_dict.fmax)

    def get_ema_features(self, sample_path):
        """Read in EMA features from csv file"""
        col_names = list(range(10))
        data = pd.read_csv(sample_path, header=None, names=col_names)
        features = torch.LongTensor(data[col_names].values).transpose(0,1)
        features = (features - self.ema_min) / (self.ema_max - self.ema_min)     # does min-max norm using local values make sense for EMA features?
        return features.to(self.device)

    def get_ema_sample_path(self, index, ema_dir):
        """get path to EMA .csv file"""
        ema_filename = str(self.ema_list.iloc[index, 0])

        return os.path.join(ema_dir, ema_filename)

    def get_mel_spectrogram(self, sample_path):
        """Load audio and return mel-spectrogram"""
        wav, sr = load_wav(sample_path)
        wav = wav / MAX_WAV_VALUE # normalise audio by largest value in dataset
        wav = normalize(wav) * 0.95
        wav = torch.FloatTensor(wav).to(self.device)
        wav = wav.unsqueeze(0)
        mel_signal = self.get_mel(wav)
        return mel_signal, sr

    def pad_signal(self, signal):
        """left-pad samples with zeros to length of longest sample in dataset"""
        signal_length = list(signal.size())[0]
        self.longest = signal_length if signal_length > self.longest else self.longest
        signal = signal.transpose(0,1)
        padder = nn.ZeroPad2d((self.max_input_length-signal_length, 0, 0, 0))
        # return torch.flatten(padder(signal)).unsqueeze(0)
        return padder(signal).unsqueeze(0)

    def get_speaker_ID(self, index, annotations):
        """get ID (index) encoding of speaker to feed into embedding layer of NN"""
        filename = annotations.iloc[index, 0]
        speaker_name = filename[:-10]
        return self.ID_dict[speaker_name]

    def get_audio_sample_path(self, index, audio_dir, annotations):
        """get path to audio wav file"""
        # print(os.path.join(audio_dir, annotations.iloc[index, 0]))
        return os.path.join(audio_dir, annotations.iloc[index, 0])

    def get_sample_ID(self, index, annotations):
        """get ID of audio file"""
        return annotations.iloc[index,0][:-4]

    def get_audio_sample_label(self, index):
        """get label [s, t] from file (not really necessary)"""
        return self.annotations.iloc[index, 1]

    def get_reference_audio(self, index):
        """NOT USED (was for MFCCS)"""
        audio_sample_path = self.get_audio_sample_path(index,
                                                        audio_dir=self.ref_audio_dir,
                                                        annotations=self.ref_annotations)
        audio_label = self.get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        mfcc_transform = MFCC(sample_rate=sr,
                                n_mfcc=40,
                                melkwargs={"n_fft": 2048, "hop_length": 512, "power": 2})
        mel_signal = mfcc_transform(torch.tensor(signal))
        return mel_signal

    def get_mfcc(self, sample_path, sr):
        """NOT USED (was for MFCCS)"""
        wav, sr = load_wav(sample_path)
        wav = torch.FloatTensor(wav).cpu()#.to(self.device)
        wav = wav.unsqueeze(0)
        mfcc_transform = MFCC(sample_rate=sr,
                                n_mfcc=40,
                                melkwargs={"n_fft": 2048, "hop_length": 256, "power": 2})
        mel_signal = mfcc_transform(torch.tensor(wav))
        return(mel_signal)

    def plot_dtw(query, reference):
        """plots sequence paths for source & warped target"""
        plt.plot(reference);
        plt.plot(query);
        plt.gca().set_title("Warping dysarthric")
        plt.show()

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    """One training epoch"""
    converged = False # becomes True when loss decrease is below threshold (latency = 5 or 10)
    loss_aggr = 0
    for inputs, speaker_ID, sr, ema_features, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.squeeze(1).to(device)
        predictions, ema_predictions = model(inputs, speaker_ID)
        print(f"input size : {list(inputs.size())}")
        print(f"target size: {list(targets.size())}")
        print(f"feature size  in train_one_epoch: {list(ema_features.size())}")
        print(f"ema prediction size  in train_one_epoch: {list(ema_predictions.size())}")
        m_s_loss = loss_fn(predictions.squeeze(0), targets)
        ema_loss = loss_fn(ema_predictions.squeeze(0), ema_features) # secondary loss term for multi-task combined loss
        loss = m_s_loss + ema_loss # sum both loss terms
        optimiser.zero_grad() # gradients get saved at each iteration --> reset gradients at each batch after update
        loss.backward()
        optimiser.step()
        loss_aggr += loss.item()
    loss_aggr /= len(data_loader.dataset)
    with open(f"2ndlog_{MODEL_NAME}.txt", "a") as o:
        o.write(f"Loss = {loss_aggr}\n")
    print(f"Loss = {loss_aggr}")
    return loss_aggr, converged

def predict(model, input, target, ema_features, speaker_ID):
    """Fix model and calculate loss on validation set (for early stopping)"""
    model.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        predicted, ema_predictions = model(input, speaker_ID) # tensor with dims: 1= input, 2=classes predicted
        m_s_loss = loss_fn(predicted.squeeze(0), target)
        ema_loss = loss_fn(ema_predictions.squeeze(0), ema_features)
        test_loss = m_s_loss + ema_loss
    return predicted, target, test_loss, ema_predictions

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    """trains, checks loss, and writes training progress to file each epoch"""
    with open(f"2ndlog_{MODEL_NAME}.txt", "w+") as o:
        pass
    previous_loss = [1000, 1000, 1000, 1000, 1000]
    best_val_loss = 1000
    for i in range(epochs):
        print(f"Epoch = {i+1}")
        with open(f"2ndlog_{MODEL_NAME}.txt", "a") as o:
            o.write(f"Epoch = {i+1}\n")
        current_loss, converged = train_one_epoch(model,
                                                data_loader,
                                                loss_fn,
                                                optimiser,
                                                device)
        if current_loss > 0.999*previous_loss[-3]:
            with open(f"2ndlog_{MODEL_NAME}.txt", "a") as o:
                o.write(f"Loss hasn't decreased by 0.0001 in 3 epochs : {previous_loss[-3]} to {current_loss}")
            print(f"Loss hasn't decreased by 0.0001% in 3 epochs : {previous_loss[-3]} to {current_loss}")
            break
        else:
            previous_loss.append(current_loss)
        loss_aggregator = 0

        for i in range(len(val_set)):
            val_inputs, val_ema,  val_targets = val_set[i][0], val_set[i][-2], val_set[i][-1]
            speaker_ID = val_set[i][1]
            val_inputs = val_inputs.unsqueeze(0)
            val_targets = val_targets.unsqueeze(0)
            predicted, expected, val_loss, ema_predictions = predict(convolutional_neural_net,
                                                    val_inputs,
                                                    val_targets,
                                                    val_ema,
                                                    speaker_ID)
            loss_aggregator += val_loss
        avg_val_loss = loss_aggregator/len(val_set)
        # if avg_val_loss > (best_val_loss*1.1):
        #     with open(f"2ndlog_{MODEL_NAME}.txt", "a") as o:
        #         o.write(f"Validation loss increased from the best {best_val_loss} to {avg_val_loss}\n")
        #     print(f"Validation loss increased from the best  {best_val_loss} to {avg_val_loss}")
        #     break
        # else:
        #     with open(f"2ndlog_{MODEL_NAME}.txt", "a") as o:
        #         o.write(f"Validation loss = {avg_val_loss}\n")
        #     print(f"Validation loss {avg_val_loss}")
        with open(f"2ndlog_{MODEL_NAME}.txt", "a") as o:
            o.write(f"Validation loss = {avg_val_loss}\n")
        print(f"Validation loss {avg_val_loss}")
        my_lr_scheduler.step()
        gc.collect()
        torch.save(convolutional_neural_net.state_dict(), f"{TRAINED_MODEL_FOLDER}/{MODEL_NAME}.pth")
        model.train()
        print("----------------------------------------------------------------------------")

    print("Training complete.")


def mean_and_sd(dataset):
    """calculate mean & s.d. for zero-mean unit-variance normalisation"""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    data = next(iter(loader))
    source_cat_target = torch.cat((data[0], data[-1]))
    mean = source_cat_target.mean()
    std = source_cat_target.std()
    return mean, std


def min_and_max(train_set, val_set):
    """calculate min & min for min-max normaliasation"""
    loader = DataLoader(train_set, batch_size=1, shuffle=True)#, num_workers=1)
    loader2 = DataLoader(val_set, batch_size=1, shuffle=True)#, num_workers=1)
    max1 = 0
    min1 = 100
    for i in range(len(train_set)):
        data = next(iter(loader))
        data = torch.cat((data[0], data[-1]))
        max1 = torch.max(data) if torch.max(data) > max1 else max1
        min1 = torch.min(data) if torch.min(data) < min1 else min1

    for i in range(len(val_set)):
        data = next(iter(loader2))
        data = torch.cat((data[0], data[-1]))
        max1 = torch.max(data) if torch.max(data) > max1 else max1
        min1 = torch.min(data) if torch.min(data) < min1 else min1


def ema_min_and_max(train_set):
    """calculate min & min EMA feature values for min-max normaliasation"""
    loader = DataLoader(train_set, batch_size=1, shuffle=True)#, num_workers=1)
    max1 = 0
    min1 = 100
    for i in range(len(train_set)):
        data = train_set[i][-2]
        print(list(data.size()))
        max1 = torch.max(data) if torch.max(data) > max1 else max1
        min1 = torch.min(data) if torch.min(data) < min1 else min1
    return min1, max1

if __name__=="__main__":

     """ load in model hyperparameters from config files"""
    converter_config_file = r"config.json"
    with open(converter_config_file) as f:
        data = f.read()
   
    global c
    json_config = json.loads(data)
    c = AttrDict(json_config)
    print(os.getcwd())
    hifigan_config_file = r"../resources/hifigan-config.json"
    if not os.path.exists(hifigan_config_file):
        hifigan_config_file = r"../hifigan-config.json"
    with open(hifigan_config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    SOURCE_ANNOTATIONS_FILE = c.source_annotations_file
    SOURCE_AUDIO_DIR = c.source_audio_dir
    TARGET_ANNOTATIONS_FILE = c.target_annotations_file
    TARGET_AUDIO_DIR = c.target_audio_dir

    TRAINED_MODEL_FOLDER = c.trained_model_folder
    if not os.path.exists(TRAINED_MODEL_FOLDER):
        pathlib.Path(TRAINED_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
        print("Made output folder to store trained model.")

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MIN_MEL = c.min
    MAX_MEL = c.max
    BATCH_SIZE = c.batch_size
    MAX_INPUT_LENGTH = c.longest
    N_MEL_FEATURES = c.n_mels
    EPOCHS = c.epochs
    LEARNING_RATE = c.learning_rate
    DECAY_RATE = c.lr_decay
    STEP_SIZE = c.step_size
    MODEL_NAME = c.model_name
    ID_DICT = c.speaker_ID_dict
    EMA_LIST = c.target_ema_file
    EMA_DIR = c.ema_dir
    EMA_MIN = c.ema_min
    EMA_MAX = c.ema_max

    global best_loss
    best_loss = 10000 # gets replaced by training loss during first epoch

    """Create and process the TORGO dataset"""

    dataset = TORGODataset(SOURCE_ANNOTATIONS_FILE,
                                                SOURCE_AUDIO_DIR,
                                                TARGET_ANNOTATIONS_FILE,
                                                TARGET_AUDIO_DIR,
                                                min_mel = MIN_MEL,
                                                max_mel = MAX_MEL,
                                                ID_dict=ID_DICT,
                                                ema_dir=EMA_DIR,
                                                ema_list=EMA_LIST,
                                                ema_min = EMA_MIN,
                                                ema_max = EMA_MAX,
                                                which_device = device,
                                                parameter_dict = h,
                                                align=True,
                                                max_input_length=MAX_INPUT_LENGTH
                                                )
    validation_split = 0.01
    shuffle_dataset = True
    random_seed= 42

    """Creating data indices for training and validation splits:"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(numpy.floor(validation_split * dataset_size))
    if shuffle_dataset :
        numpy.random.seed(random_seed)
        numpy.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    """Creating Torch data samplers and loaders:"""
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_data_loader = DataLoader(dataset,
                                    batch_size=BATCH_SIZE,
                                    sampler=train_sampler)
    val_data_loader = DataLoader(dataset,
                                    batch_size=len(val_indices),
                                    sampler=valid_sampler)
    val_set = Subset(dataset, val_indices)

    """Instantiate model (load from file if exists) and train"""
    convolutional_neural_net = ConvNet2D().to(device)
    if os.path.exists(f"../trained_models/{MODEL_NAME}.pth"):
        state_dict = torch.load(f"../trained_models/{MODEL_NAME}.pth",  map_location=device)
        convolutional_neural_net.load_state_dict(state_dict)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(convolutional_neural_net.parameters(),
                                lr=LEARNING_RATE)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, gamma=DECAY_RATE, step_size=STEP_SIZE)
    train(convolutional_neural_net, train_data_loader, loss_fn, optimiser, device, epochs=EPOCHS)
    torch.save(convolutional_neural_net.state_dict(), f"{TRAINED_MODEL_FOLDER}/{MODEL_NAME}.pth")
    print(f"Model trained and stored at /msc-project-voice-conversion/trained_models/{MODEL_NAME}.pth")