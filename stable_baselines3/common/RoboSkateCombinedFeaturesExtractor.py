import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RoboSkateCombinedFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=32):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(RoboSkateCombinedFeaturesExtractor, self).__init__(observation_space, features_dim=32)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # get imput layer (mostly RGB)
                n_input_channels = subspace.shape[0]
                #define CNN Layer
                # small CNN version
                smallcnn = nn.Sequential(nn.Conv2d(n_input_channels, 32, kernel_size=6, stride=3),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                         nn.ReLU()
                                         )
                # large CNN version
                largecnn = nn.Sequential(   nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 256, kernel_size=4, stride=2),
                                            nn.ReLU()
                                            )

                self.cnn = smallcnn

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = self.cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                print("CNN output size before linear: " + str(n_flatten))

                # define Linear layer after CNN
                self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                            nn.ReLU(),
                                            )

                extractors[key] = nn.Sequential(self.cnn, self.linear)
                # Set the feature dimention as the CNN latent space so the feature dimesnon will be lager (+numeric)
                total_concat_size += features_dim

            elif key == "numeric":
                # Run through a simple MLP
                extractors[key] = nn.Identity() #nn.Linear(subspace.shape[0], 16)
                total_concat_size += 14
                # TODO: Manual set number of numerical features

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)