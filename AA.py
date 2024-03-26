class AA(nn.Module):
    def __init__(self, in_channels, out_channels, n=4, e=0.5):
        super().__init__()

        extra_branch_steps = 1
        while extra_branch_steps * 2 < n:
            extra_branch_steps *= 2
        n_list = [0, extra_branch_steps, n]
        branch_num = len(n_list)

        c_ = int(out_channels * e)  # hidden channels
        self.c = c_
        self.cv1 = Conv(in_channels, branch_num * self.c, 1, 1)
        self.cv2 = Conv((sum(n_list) + branch_num) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList()
        for n_list_i in n_list[1:]:
            self.m.append(nn.Sequential(*(Conv(self.c, self.c, 3, 1) for _ in range(n_list_i))))
        self.split_num = tuple([self.c] * branch_num)


    def forward(self, x):
        x = torch.cat(x, 1)
        y = list(self.cv1(x).split(self.split_num, 1))
        all_y = [y[0]]
        for m_idx, m_i in enumerate(self.m):
            all_y.append(y[m_idx + 1])
            all_y.extend(m(all_y[-1]) for m in m_i)
        return self.cv2(torch.cat(all_y, 1))
