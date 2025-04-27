import gradio as gr
import torch
import numpy as np
import torch.nn as nn

# import your model class and helpers
from utils import string_to_int, int_to_string, load_dataset

# model
class RepeatVector(nn.Module):
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

    def forward(self, x):
        # x: (batch_size, features)
        return x.unsqueeze(1).repeat(1, self.n, 1)
class modelf(nn.Module):
    def __init__(self, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        super(modelf, self).__init__()

        self.Tx = Tx
        self.Ty = Ty
        self.n_a = n_a
        self.n_s = n_s
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size

        # one_step_attention layers:
        self.repeator = RepeatVector(self.Tx)
        repeator = RepeatVector(self.Tx)
        self.linear1 = nn.Linear((2 * self.n_a) + self.n_s, 10)
        self.linear2 = nn.Linear(10, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # forward layers
        self.pre_attention_lstm = nn.LSTM(input_size=self.human_vocab_size, hidden_size=self.n_a, batch_first=True,
                                          bidirectional=True)
        self.post_attention_lstm = nn.LSTM(2 * self.n_a, self.n_s, batch_first=True)
        self.output_layer = nn.Linear(self.n_s, self.machine_vocab_size)
        self.softmax_main = nn.Softmax(dim=-1)

    def one_step_attention(self, a, s_prev):
        # this attention mechnism for lstm

        # add's dimenison 1 making (1,2) to (1, 3, 2)
        s_prev = self.repeator(s_prev)
        concat = torch.cat([a, s_prev], dim=-1)

        e = self.linear1(concat)
        e = self.tanh(e)

        energies = self.linear2(e)
        energies = self.relu(energies)
        # softmax on dimension 1
        alphas = self.softmax(energies)
        # this is dot product
        context = torch.sum(alphas * a, dim=1, keepdim=True)

        return context

    def forward(self, X):
        outputs = []

        a, _ = self.pre_attention_lstm(X)

        batch_size = X.shape[0]
        s = torch.zeros(batch_size, n_s).to(device)
        c = torch.zeros(batch_size, n_s).to(device)

        for t in range(self.Ty):
            # one setp attention
            context = self.one_step_attention(a, s)

            _, (s, c) = self.post_attention_lstm(context, (s.unsqueeze(0), c.unsqueeze(0)))
            s, c = s.squeeze(0), c.squeeze(0)

            out = self.output_layer(s)

            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs


# --- Load vocabularies and model ---
Tx = 30
Ty = 10
n_a = 32
n_s = 64

_, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.load_state_dict(torch.load("models/model_weights_1600.pth", map_location=device))
model.to(device)
model.eval()

# --- Translation function ---
def translate_date(human_readable_date):
    x_enc = string_to_int(human_readable_date, Tx, human_vocab)
    x_enc = np.array(list(map(lambda x: np.eye(len(human_vocab))[x], x_enc)))
    X_tensor = torch.tensor(x_enc, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        if isinstance(outputs, list):
            outputs = torch.stack(outputs, dim=1)
        preds = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()
    machine_readable_date = ''.join(int_to_string(preds, inv_machine_vocab))
    return machine_readable_date

# --- Gradio Interface ---
demo = gr.Interface(
    fn=translate_date,
    inputs=gr.Textbox(lines=1, placeholder="Enter a date (e.g. 3 May 1979)"),
    outputs="text",
    title="Neural Machine Translation - Date Format Converter",
    description="Enter a human-readable date and get the machine-readable ISO date."
)

demo.launch()
