from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        super().__init__()
        self.w = Parameter(ones((1, dimensions)))


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        return tensordot(x, self.w, dims=([-1], [-1])).reshape(-1)

        

    def get_prediction(self, x):
        return 1 if self(x).item() >= 0 else -1



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        super().__init__()
        self.first = Linear(1, 128)
        self.second = Linear(128, 128)
        self.output = Linear(128, 1)
   

    def forward(self, x):
        return self.output(relu(self.second(relu(self.first(x)))))


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        super().__init__()
        self.first = Linear(784, 256)
        self.second = Linear(256, 128)
        self.output = Linear(128, 10)


    def forward(self, x):
        return self.output(relu(self.second(relu(self.first(x)))))



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        super().__init__()
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.input = Linear(self.num_chars, 256)
        self.recurrent = Linear(256, 256, bias=False)
        self.output = Linear(256, len(self.languages))
        # Initialize your model parameters here



    def forward(self, xs):
        hidden = xs[0].new_zeros((xs[0].shape[0], 256))
        for character in xs:
            candidate = relu(self.input(character) + self.recurrent(hidden))
            present = character.sum(dim=1, keepdim=True)
            hidden = present * candidate + (1 - present) * hidden
        return self.output(hidden)



def Convolve(input: tensor, weight: tensor):
    height = input.shape[0] - weight.shape[0] + 1
    width = input.shape[1] - weight.shape[1] + 1
    if height <= 0 or width <= 0:
        raise ValueError('Kernel is larger than the input')
    # Unfold creates strided windows; tensordot performs valid cross-correlation.
    windows = input.unfold(0, weight.shape[0], 1).unfold(1, weight.shape[1], 1)
    return tensordot(windows, weight, dims=([2, 3], [0, 1]))


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """

    def __init__(self):
        super().__init__()
        self.convolution_weights = Parameter(ones((3, 3)) / 9)
        self.first = Linear(26 * 26, 128)
        self.output = Linear(128, 10)


    def forward(self, x):
        images = x.reshape(len(x), 28, 28)
        features = stack([Convolve(image, self.convolution_weights) for image in images]).flatten(start_dim=1)
        return self.output(relu(self.first(features)))


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        _, length, _ = input.size()
        scores = matmul(self.q_layer(input), movedim(self.k_layer(input), 1, 2)) / self.layer_size ** 0.5
        # The supplied 4D mask broadcasts to [1,B,T,T]; remove that leading axis.
        scores = scores.masked_fill(self.mask[:, :, :length, :length] == 0, float('-inf'))[0]
        return matmul(softmax(scores, dim=-1), self.v_layer(input))

