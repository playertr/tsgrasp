
import torch
def transform(m: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform the batched vector or pose matrix `m` by batch transform `tf`.

    Args:
        m (torch.Tensor): (b, ..., 3) or (b, ..., 4, 4) vector or pose.
        tf (torch.Tensor): (b, 4, 4) batched homogeneous transform matrix

    Returns:
        torch.Tensor: (b, ..., 3) or (b, ..., 4, 4) transformed poses or vectors.
    """
    # TODO: refactor batch handling by reshaping the arguments to transform_vec
    assert m.shape[0] == tf.shape[0], "Arguments must have same batch dimension."
    if m.shape[-1] == 3:
        return transform_vec(m, tf)
    elif m.shape[-2:] == (4, 4):
        return transform_mat(m, tf)

def transform_mat(pose: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform homogenous transformation `pose` by homogenous transformation `tf`.

    Args:
        x (torch.Tensor): (b, ..., 4, 4) homogeneous pose matrix
        tf (torch.Tensor): (b, 4, 4) homogeneous transform matrix

    Returns:
        torch.Tensor: (b, ..., 4, 4) transformed poses.
    """

    assert all((
        len(tf.shape)==3,           # tf must be (b, 4, 4)
        tf.shape[1:]==(4, 4),       # tf must be (b, 4, 4)
        tf.shape[0]==pose.shape[0], # batch dimension must be same
    )), "Argument shapes are unsupported."
    x_dim = len(pose.shape)
    
    # Pad the dimension of tf for broadcasting.
    # E.g., if pose had shape (2, 3, 7, 4, 4), and tf had shape (2, 4, 4), then
    # we reshape tf to (2, 1, 1, 4, 4)
    tf = tf.reshape(tf.shape[0], *([1]*(x_dim-3)), 4, 4)

    return tf @ pose

def transform_vec(x: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform 3D vector `x` by homogenous transformation `tf`.

    Args:
        x (torch.Tensor): (b, ..., 3) coordinates in R3
        tf (torch.Tensor): (b, 4, 4) homogeneous pose matrix

    Returns:
        torch.Tensor: (b, ..., 3) coordinates of transformed vectors.
    """

    x_dim = len(x.shape)
    assert all((
        len(tf.shape)==3,           # tf must be (b, 4, 4)
        tf.shape[1:]==(4, 4),       # tf must be (b, 4, 4)
        tf.shape[0]==x.shape[0],    # batch dimension must be same
        x_dim > 2                   # x must be a batched matrix/tensor
    )), "Argument shapes are unsupported."

    x_homog = torch.cat(
        [x, torch.ones(*x.shape[:-1], 1, device=x.device)], 
        dim=-1
    )
    
    # Pad the dimension of tf for broadcasting.
    # E.g., if x had shape (2, 3, 7, 3), and tf had shape (2, 4, 4), then
    # we reshape tf to (2, 1, 1, 4, 4)
    tf = tf.reshape(tf.shape[0], *([1]*(x_dim-3)), 4, 4)

    return (x_homog @ tf.transpose(-2, -1))[..., :3]