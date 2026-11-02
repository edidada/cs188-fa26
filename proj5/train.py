from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        while True:
            mistakes = 0
            for batch in dataloader:
                x, label = batch['x'], batch['label'].item()
                if model.get_prediction(x) != label:
                    model.w.add_(label * x)
                    mistakes += 1
            if mistakes == 0:
                return


def train_regression(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    for epoch in range(4000):
        for batch in loader:
            optimizer.zero_grad()
            loss = regression_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print('regression', epoch, loss.item(), flush=True)
        if loss.item() < 0.003:
            return


def train_digitclassifier(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(15):
        for batch in loader:
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()
        print('digits', epoch, accuracy, flush=True)
        if accuracy >= 0.981:
            return


def train_languageid(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Individual items are zero-padded by the provided dataset to length 10.
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(35):
        for batch in loader:
            optimizer.zero_grad()
            xs = movedim(batch['x'], 0, 1)
            loss = languageid_loss(model(xs), batch['label'])
            loss.backward()
            # Avoid recurrent-gradient explosions using in-place clipping.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.clamp_(-5, 5)
            optimizer.step()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()
        print('language', epoch, accuracy, flush=True)
        if accuracy >= 0.85:
            return



def Train_DigitConvolution(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    for epoch in range(80):
        for batch in loader:
            optimizer.zero_grad()
            loss = digitconvolution_Loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('convolution', epoch, loss.item(), flush=True)
