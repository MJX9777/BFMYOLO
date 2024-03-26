class BFM(nn.Module):  # stereo attention block
    def __init__(self, channels):
        super(BFM, self).__init__()
        hid_c = 32
        self.sa_ir = ESA(1, hid_c, 3)
        self.sa_rgb = ESA(channels, hid_c, 3)
        self.cat = Concat()
        self.fc = nn.Linear(hid_c * 2, hid_c // 4)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(hid_c // 4, hid_c))
        self.softmax = nn.Softmax(dim=1)
        self.cv_e = Conv(int(hid_c * 2), 64)


    def forward(self, x):
        x_rgb_ori, x_ir_ori = x[0], x[1]
        x1 = self.sa_rgb(x_rgb_ori)
        x2 = self.sa_ir(x_ir_ori)

        x = self.cat([x1, x2])
        x = self.fc(x.mean(-1).mean(-1))

        for i, fc in enumerate(self.fcs):
            vector = fc(x).unsqueeze(dim=1)
            if i == 0:
                scores = vector
            else:
                scores = torch.cat([scores, vector], dim=1)

        scores = self.softmax(scores).unsqueeze(-1).unsqueeze(-1)

        a1, a2 = scores[:, 0, ...], scores[:, 1, ...]
        # x = a1 * x1 + a2 * x2
        x = self.cat([a1 * x1, a2 * x2])
        return self.cv_e(x)
