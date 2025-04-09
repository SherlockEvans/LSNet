class I-FocalLoss(nn.Module):

    def __init__(self, alpha=0.1, gamma=1): # 定义alpha和gamma变量
        super(BinaryFocalLoss3, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    # 前向传播
    def forward(self, preds, labels):
        eps = 1e-7  # 防止数值超出定义域
        preds = F.softmax(preds, dim=-1)
        # preds = torch.sigmoid(preds)
        labels = labels.view(-1, 1)
        print("labels:",labels)
        print(labels.shape)
        # 开始
        # log = torch.log(preds)
        # print("log:",log)

        
        loss_y1 = -1 * self.alpha * 2.718 * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        # print("loss_y1:",loss_y1)

        loss_y0 = -1 * (1-self.alpha) * 2.718 * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        # print("loss_y0:", loss_y0)

        loss = loss_y0 + loss_y1
        # print("loss:", loss)
        
        return torch.mean(loss, dim=0, keepdim=True)[0, 1]
