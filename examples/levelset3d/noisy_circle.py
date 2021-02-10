from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
from topologylayer.nn import LevelSetLayer3D, SumBarcodeLengths, PartialSumBarcodeLengths
import torch
import torch.nn as nn

# generate circle on grid
n = 16
def circlefn(i, j, k, n):
    r = np.sqrt((i - n/2.)**2 + (j - n/2.)**2)
    return np.exp(-((r - n/3.)**2 + (k - n/2.)**2)/(n*2))


def gen_circle(n):
    beta = np.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                beta[i, j, k] = circlefn(i, j, k, n)
    return beta

beta = gen_circle(n)

m = 1500
X = np.random.randn(m, n**3)
y = X.dot(beta.flatten()) + 0.05*np.random.randn(m)
beta_ols = (np.linalg.lstsq(X, y, rcond=None)[0]).reshape(n, n, n)


class TopLoss(nn.Module):
    def __init__(self, size):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer3D(size=size, sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=1)
        self.topfn2 = SumBarcodeLengths(dim=0)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo) + self.topfn2(dgminfo)


tloss = TopLoss((n, n, n))  # topology penalty
dloss = nn.MSELoss()  # data loss

beta_t = torch.autograd.Variable(torch.tensor(beta_ols).type(torch.float), requires_grad=True)
X_t = torch.tensor(X, dtype=torch.float, requires_grad=False)
y_t = torch.tensor(y, dtype=torch.float, requires_grad=False)
optimizer = torch.optim.Adam([beta_t], lr=1e-2)
fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
ax[0].imshow(np.mean(beta, axis=-1))
ax[0].set_title("Truth")
ax[1].imshow(np.mean(beta_ols, axis=-1))
ax[1].set_title("OLS")
ax[2].set_title("Topology Regularization")
for i in range(500):
    optimizer.zero_grad()
    tlossi = tloss(beta_t)
    dlossi = dloss(y_t, torch.matmul(X_t, beta_t.view(-1)))
    loss = 0.1*tlossi + dlossi
    loss.backward()
    optimizer.step()
    # if (i % 10 == 0):
    print(i, tlossi.item(), dlossi.item())
    beta_est = beta_t.detach().numpy()
    ax[2].imshow(np.mean(beta_est, axis=-1))
    for i in range(3):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].tick_params(bottom=False, left=False)
    plt.pause(0.05)

beta_est = beta_t.detach().numpy()
# save figure
# plt.savefig('noisy_circle.png')
img = nib.Nifti1Image(beta, np.eye(4))
nib.save(img, 'truth.nii.gz')
img = nib.Nifti1Image(beta_ols, np.eye(4))
nib.save(img, 'ols.nii.gz')
img = nib.Nifti1Image(beta_est, np.eye(4))
nib.save(img, 'topreg.nii.gz')
