import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(384, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D),
                                        nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D),
                                        nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2)
        )

    def forward(self, x):

        m=nn.BatchNorm1d(x.shape[1])
        m.cuda()
        x=m(x)
        H = self.feature_extractor_part2(x)  # NxL
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        H=H.squeeze(0)
        A=A.view(A.size(0), -1)
        A=A.transpose(0,1)
        M = torch.mm(A, H)  # KxL
        M1 = self.attention(M)
        M2 = M1*M

        Y_prob = self.classifier(M2)

        return M2.squeeze(0),Y_prob

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
