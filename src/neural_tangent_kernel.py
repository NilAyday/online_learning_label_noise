import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if x.requires_grad:
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad
    return None


def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in[1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[1], *x.shape)
    for i in range(y.shape[1]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[0][i] = 1
        jac[i] = gradient(y, x, grad_outputs=grad_outputs)
        del grad_outputs
    return jac


def downsample(nth, a):
    """
        keep every nth element
    """
    m = a.size(0)
    a = a[:int(m / nth) * nth]
    return a.reshape(-1, nth)[:, :1].reshape(-1)


def collect_gradient_embedding(model_fn, model, inp, threshold=40009, cutoff=0):
    """
        Computes the ntks embedding normally
    """
    y = model_fn(inp, model)
    ntk = torch.tensor([])
    for i_param, param in enumerate(model.parameters()):
        if i_param >= cutoff:
            if param.requires_grad:
                try:
                    jac = jacobian(y, param)
                    jac = torch.flatten(jac)
                    if jac.size(0) > threshold:
                        jac = downsample(int(jac.size(0) / threshold), jac)
                    ntk = torch.cat((ntk, jac))
                except:
                    pass
    return ntk.view(1, -1)

def neural_tangent_kernel_matrix_precomputed(labels, y, model, inputs, threshold=40009, cutoff=0, every_nth=1):
    ntk = torch.zeros(len(inputs), len(inputs))
    norms = torch.zeros(len(inputs))

    for i_param, param in enumerate(model.parameters()):
        if i_param >= cutoff:
            if (i_param + 1) % every_nth == 0:
                if param.requires_grad:
                    grads = []
                    for i, _ in enumerate(inputs):
                        jac = jacobian(y[i], param)  # jacobian of current logit
                        jac = torch.flatten(jac)
                        if jac.size(0) > threshold:
                            jac = downsample(int(jac.size(0) / threshold), jac)
                        grads.append(jac.cpu().detach())

                        del jac

                    grads = torch.stack(grads)
                    grads = torch.matmul(grads, grads.T)
                    ntk += grads

                    norm_ = torch.sum(grads * grads, dim=1)
                    norms += norm_

                    del grads
                    del norm_

    norms = torch.sqrt(norms)
    norms = torch.outer(norms, norms)
    norms[norms < 1e-08] = 1e-08

    return ntk, norms, labels

def neural_tangent_kernel_matrix(model_fn, model, inputs, threshold=40009, cutoff=0, every_nth=1):
    """
    compute ntk kernel matrix layer wise
        downsample if needed to not kill the computer
        Threshold should be prime to avoid sampling unfairly
    Parameters
    ----------
    model_fn
    model
    inputs
    threshold
    cutoff
    every_nth
        Allows to only compute the gradients of every nth layer

    Returns
    -------

    """
    model = model.to(device)

    labels = []
    y = []
    for inp in inputs:
        logits = model_fn(inp, model).cpu()
        y.append(logits)
        labels.append(torch.argmax(logits, dim=1).item())

    return neural_tangent_kernel_matrix_precomputed(labels, y, model, inputs, threshold, cutoff, every_nth)


def cosine_distances(ntk, norms):
    """
        Helper function to convert ntk and norms to distances for TSNE
    """
    ntk = ntk / norms

    ntk *= -1
    ntk += 1
    ntk[ntk < 0] = 0.
    return ntk
