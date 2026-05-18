import torch
import gpytorch

# We will use the simplest form of GP model, exact inference
class SMF_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SMF_Model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))

    def initialize(self):
        def inv_softplus(x):
            x = torch.as_tensor(x, dtype=torch.float)
            return torch.log(torch.expm1(x))
        
        with torch.no_grad():
            # Lengthscales: random in [0.5, 1.5]
            ls = 0.5 + torch.rand_like(self.covar_module.base_kernel.raw_lengthscale)
            self.covar_module.base_kernel.raw_lengthscale.copy_(inv_softplus(ls))
            
            # Outputscale: random in [0.5, 2.0]
            os_val = 0.5 + 1.5 * torch.rand(()).item()
            self.covar_module.raw_outputscale.fill_(inv_softplus(os_val).item())
            
            # Mean: small random
            self.mean_module.raw_constant.fill_((torch.randn(()) * 0.1).item())
               
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class fgas_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(fgas_Model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))

    def initialize(self):
        def inv_softplus(x):
            x = torch.as_tensor(x, dtype=torch.float)
            return torch.log(torch.expm1(x))
        
        with torch.no_grad():
            # Lengthscales: random in [0.5, 1.5]
            ls = 0.5 + torch.rand_like(self.covar_module.base_kernel.raw_lengthscale)
            self.covar_module.base_kernel.raw_lengthscale.copy_(inv_softplus(ls))
            
            # Outputscale: random in [0.5, 2.0]
            os_val = 0.5 + 1.5 * torch.rand(()).item()
            self.covar_module.raw_outputscale.fill_(inv_softplus(os_val).item())
            
            # Mean: small random
            self.mean_module.raw_constant.fill_((torch.randn(()) * 0.1).item())

        # reset outputscale
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SMHM_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SMHM_Model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))

    def initialize(self):
        def inv_softplus(x):
            x = torch.as_tensor(x, dtype=torch.float)
            return torch.log(torch.expm1(x))
        
        with torch.no_grad():
            # Lengthscales: random in [0.5, 1.5]
            ls = 0.5 + torch.rand_like(self.covar_module.base_kernel.raw_lengthscale)
            self.covar_module.base_kernel.raw_lengthscale.copy_(inv_softplus(ls))
            
            # Outputscale: random in [0.5, 2.0]
            os_val = 0.5 + 1.5 * torch.rand(()).item()
            self.covar_module.raw_outputscale.fill_(inv_softplus(os_val).item())
            
            # Mean: small random
            self.mean_module.raw_constant.fill_((torch.randn(()) * 0.1).item())

        # reset outputscale
        self.covar_module.outputscale = 1.0
        self.likelihood.noise = 1e-2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class profile_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(profile_Model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))

    def initialize(self):
        # reset constant mean to something reasonable
        self.mean_module.constant.data.normal_(0, 0.1)

        # lengthscales (ARD) :
        # initialize in log-space so we get spread over ~[0.3, 3] in standardized units
        with torch.no_grad():
            ls = 0.5 + torch.rand_like(self.covar_module.base_kernel.lengthscale)
            self.covar_module.base_kernel.lengthscale.copy_(ls)

        # reset outputscale
        self.covar_module.outputscale = 1.0

        # reset likelihood noise to moderate value
        self.likelihood.noise = 0.05

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BHMF_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(BHMF_Model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))

    def initialize(self):
        # reset constant mean to something reasonable
        self.mean_module.constant.data.normal_(0, 0.1)

        # lengthscales (ARD) :
        # initialize in log-space so we get spread over ~[0.3, 3] in standardized units
        with torch.no_grad():
            ls = 0.5 + torch.rand_like(self.covar_module.base_kernel.lengthscale)
            self.covar_module.base_kernel.lengthscale.copy_(ls)

        # reset outputscale
        self.covar_module.outputscale = 1.0

        # reset likelihood noise to moderate value
        self.likelihood.noise = 0.05

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
