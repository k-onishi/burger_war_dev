import torch
from torch import nn
from torch.nn import functional as F

# respect kenjirotorii
class MaskNet(nn.Module):
    def __init__(self, num_outputs, duel=True):
        super(MaskNet, self).__init__()
        self.duel = duel
        '''
        # input state
        self.state = {
            "pose": self.pose,              # (N, 2)
            "lidar": self.lidar_ranges,     # (N, 1, 360)
            "image": self.image,            # (N, 3, 480, 640)
            "mask": self.mask,              # (N, 18)
        }
        '''
        # OpenAI: Emergent Tool Use from Multi-Agent Interaction
        # https://openai.com/blog/emergent-tool-use/
        # https://pira-nino.hatenablog.com/entry/introduce_openai_hide-and-seek

        # Core network
        self.block_lidar = nn.Sequential(
            # Input size: (1, 1, 360)
            nn.Conv1d(1, 16, 3, padding=2, padding_mode='circular'),    # (N, 16, 360)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=6),                                # (N, 16, 60)
            nn.Conv1d(16, 32, 3, padding=2, padding_mode='circular'),   # (N, 32, 60)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),                                # (N, 32, 20)
            nn.Conv1d(32, 64, 3, padding=2, padding_mode='circular'),   # (N, 64, 20)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*20, 64)
        )

        self.block_image = nn.Sequential(
            # Input size: (1, 3, 95, 160)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (N, 16, 95, 160)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),                 # (N, 16, 23, 40)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (N, 32, 23, 40)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),                 # (N, 32, 5, 10)
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (N, 32, 5, 10)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32*5*10, 64)
        )
        
        # middle
        self.fc1 = nn.Sequential(
            nn.Linear(130, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 18, kernel_size=3, padding=1),
            nn.BatchNorm1d(18),
            nn.ReLU(inplace=True),
        )
        self.mask_fc = nn.Sequential(
            nn.Linear(18, 18),
            nn.ReLU(inplace=True),
        )
        # head
        self.fc2 = nn.Linear(64, num_outputs)

        # Dueling network
        self.fc_adv = nn.Linear(64, num_outputs)
        self.fc_val = nn.Linear(64, 1)

    def forward(self, pose, lidar, image, mask):
        # Core network
        ## Process each input
        x = self.block_lidar(lidar)     # (N, 64)
        y = self.block_image(image)     # (N, 64)

        ## Merge intermediate results
        w = torch.cat([pose, x, y], dim=1)    # (N, 130)

        ## Middle
        w = self.fc1(w)
        w = w.view(-1, 3, 64)       # (N, 3, 64)
        w = self.conv1(w)           # (N, 18, 64)

        ## Mask
        m = self.mask_fc(mask)      # (N, 18)
        m = m.view(-1, 1, 18)       # (N, 1, 18)
        w = torch.matmul(m, w)      # (N, 1, 64)
        w = w.view(-1, 64)          # (N, 64)

        ## Head
        if not self.duel:
            w = self.fc2(w)
        else:
            # Dueling network
            adv = self.fc_adv(w)
            val = self.fc_val(w).expand(-1, adv.size(1))
            w = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return w

