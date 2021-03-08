import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskNet(nn.Module):
    def __init__(self, output_size, duel=False):
        """
        Args:
            output_size (int): size of output
        """
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
            # Input size: (1, 3, 480, 640)
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # (N, 16, 238, 318)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),                # (N, 16, 79, 106)
            nn.Conv2d(16, 32, kernel_size=5, stride=2), # (N, 32, 38, 51)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                # (N, 32, 19, 25)
            nn.Conv2d(32, 32, kernel_size=5, stride=2), # (N, 32, 8, 11)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32*8*11, 64)
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
        self.fc2 = nn.Linear(64, output_size)
    
        # Dueling network
        self.fc_adv = nn.Linear(64, output_size)
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
        mask = self.mask_fc(mask)           # (N, 18)
        mask = mask.view(-1, 1, 18)         # (N, 1, 18)
        w = torch.matmul(mask, w)           # (N, 1, 64)
        w = w.view(-1, 64)                  # (N, 64)

        ## Head
        if not self.duel:
            w = self.fc2(w)
        else:
            # Dueling network
            adv = self.fc_adv(w)
            val = self.fc_val(w).expand(-1, adv.size(1))
            w = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return w


if __name__ == '__main__':
    def example(net, device_name):
        # Prepare sample datasets
        device = torch.device(device_name)
        net = net.to(device)
        pose = torch.randn(data_sizes[0]).to(device)
        lidar = torch.randn(data_sizes[1]).to(device)
        image = torch.randn(data_sizes[2]).to(device)
        mask = torch.randn(data_sizes[3]).to(device)

        # Run
        import time
        print('[{}] Processing...'.format(device))
        start_time = time.time()
        val = net(pose, lidar, image, mask)
        elapsed_time = time.time() - start_time
        print('[{}] Done. {:.3f}[ms]'.format(device, elapsed_time))
        print(val[0])

    net = MaskNet(5)

    # Summarize
    #from torchinfo import summary
    data_sizes = [
        (2, 2),
        (2, 1, 360),
        (2, 3, 480, 640),
        (2, 18),
    ]
    #summary(net, data_sizes)
    
    # Test run
    for _ in range(3):
        example(net, 'cpu')
    if torch.cuda.is_available():
        example(net, 'cuda:0')
    else:
        print('* CUDA not available.')
