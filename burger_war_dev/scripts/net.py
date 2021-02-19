import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int): size of output
        """
        super(Net, self).__init__()

        '''
        # input state
        self.state = {
            "lidar": self.lidar_ranges,     # (1, 360)
            "map": map,                     # (1, 2, 16, 16)
            "image": self.image             # (1, 3, 480, 640)
        }
        '''
        # OpenAI: Emergent Tool Use from Multi-Agent Interaction
        # https://openai.com/blog/emergent-tool-use/
        # https://pira-nino.hatenablog.com/entry/introduce_openai_hide-and-seek

        # Core network
        self.block_lidar = nn.Sequential(
            # TODO: UPDATE ME!
            nn.Conv1d(1, 16, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv1d(16, 16, 3, padding=1, padding_mode='circular')
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_map = nn.Sequential(
            # TODO: UPDATE ME!
            nn.Conv2d(2, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, padding=1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_image = nn.Sequential(
            # TODO: UPDATE ME!
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_fc = nn.Sequential(
            # TODO: UPDATE ME!
            nn.Linear(512 + 4 * 4 + 20 * 15, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64)
        )
        
        # Dueling network
        self.fc_adv = nn.Linear(64, output_size)
        self.fc_val = nn.Linear(64, 1)

    def forward(self, lidar, map, image):
        # Core network
        x = self.block_lidar(lidar)
        y = self.block_map(map)
        z = self.block_image(image)
        y = torch.flatten(y, 1)
        z = torch.flatten(z, 1)
        x = torch.cat([x, y, z], dim=1)
        x = self.block_fc(x)

        # Dueling network
        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(-1, adv.size(1))
        x = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return x

if __name__ == '__main__':
    net = Net(5)
    print(net)
