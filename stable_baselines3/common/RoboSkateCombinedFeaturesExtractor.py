# This feature extractor is used to combine image data and numerical observations from RoboSkate as a common input.
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
                self.cnn = nn.Sequential(   # ((in+2*pad-Kern)/Str)+1
                                            # 3 x 30 x 98
                                            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            # 16 x 14 x 48
                                            nn.Conv2d(16, 32, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            # 32 x 6 x 23
                                            nn.Conv2d(32, 64, kernel_size=[3,3], stride=[1,2]),
                                            nn.ReLU(),
                                            # 64 x 4 x 11 = 2816
                                            nn.Flatten()
                                         )


                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = self.cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                print("Flatten CNN output size: " + str(n_flatten))

                # define Linear layer after CNN
                self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                            nn.ReLU(),
                                            )
                print(n_flatten)
                print(features_dim)
                extractors[key] = nn.Sequential(self.cnn, self.linear)
                # Set the feature dimention as the CNN latent space so the feature dimesnon will be lager (+numeric)
                total_concat_size += features_dim

            elif key == "numeric":
                # Run through a simple MLP
                extractors[key] = nn.Identity() #nn.Linear(subspace.shape[0], 16)
                total_concat_size += 11
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