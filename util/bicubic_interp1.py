import numpy as np
import torch
from torch.autograd import Variable

def bicubic_interp_2d(input_, new_size, endpoint=False):
  """
  Args :
    input_ : Input tensor. Its shape should be
        [batch, channel, height, width].
        In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
  ref : 
    https://github.com/iwyoo/bicubic_interp-tensorflow
  """

  shape = input_.shape
  batch = shape[0]
  channel = shape[1]
  height  = shape[2]
  width   = shape[3]
 
  def _hermite(A, B, C, D, t):
    a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
    b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
    c = A * (-0.5) + C * 0.5
    d = B

    return a*t*t*t + b*t*t + c*t + d

  def _get_grid_array(n_i, c_i, y_i, x_i):
    n, c, y, x = np.meshgrid(n_i, c_i, y_i, x_i, indexing='ij')
    n = np.expand_dims(n, axis=4)
    c = np.expand_dims(c, axis=4)
    y = np.expand_dims(y, axis=4)
    x = np.expand_dims(x, axis=4)
    
    return np.concatenate([n,c,y,x], axis=4)

  def _get_frac_array(n, c, y_d, x_d):
    y = y_d.shape[0]
    x = x_d.shape[0]
    y_t = y_d.reshape([1, 1, -1, 1])
    x_t = x_d.reshape([1, 1, 1, -1])
    y_t = np.tile(y_t, (n,c,1,x))
    x_t = np.tile(x_t, (n,c,y,1))
#    y_t.dtype = 'float'
    y_t = Variable(torch.from_numpy(y_t).float().cuda(), volatile=False)
#    x_t.dtype = 'float'
    x_t = Variable(torch.from_numpy(x_t).float().cuda(), volatile=False)
    return y_t, x_t

  def _get_index_tensor(grid, x, y):
    new_grid = np.array(grid)

    grid_y = grid[:,:,:,:,2] + y
    grid_x = grid[:,:,:,:,3] + x

    grid_y = np.clip(grid_y, 0, height-1)
    grid_x = np.clip(grid_x, 0, width-1)

    new_grid[:,:,:,:,2] = grid_y
    new_grid[:,:,:,:,3] = grid_x
    new_grid.dtype = 'int'

    return new_grid

  new_height = new_size[0]
  new_width  = new_size[1]

  n_i = np.arange(batch)
  c_i = np.arange(channel)

  if endpoint:
    y_f = np.linspace(0., height-1, new_height)
  else:
    y_f = np.linspace(0., height, new_height, endpoint=False)
  y_i = y_f.astype(np.int32)
  y_d = y_f - np.floor(y_f)

  if endpoint:
    x_f = np.linspace(0., width-1, new_width)
  else:
    x_f = np.linspace(0., width, new_width, endpoint=False)
  x_i = x_f.astype(np.int32)
  x_d = x_f - np.floor(x_f) 

  grid = _get_grid_array(n_i, c_i, y_i, x_i)
  y_t, x_t = _get_frac_array(batch, channel, y_d, x_d)

  i_00 = _get_index_tensor(grid, -1, -1)
  i_10 = _get_index_tensor(grid, +0, -1)
  i_20 = _get_index_tensor(grid, +1, -1)
  i_30 = _get_index_tensor(grid, +2, -1)
      
  i_01 = _get_index_tensor(grid, -1, +0)
  i_11 = _get_index_tensor(grid, +0, +0)
  i_21 = _get_index_tensor(grid, +1, +0)
  i_31 = _get_index_tensor(grid, +2, +0)
      
  i_02 = _get_index_tensor(grid, -1, +1)
  i_12 = _get_index_tensor(grid, +0, +1)
  i_22 = _get_index_tensor(grid, +1, +1)
  i_32 = _get_index_tensor(grid, +2, +1)
      
  i_03 = _get_index_tensor(grid, -1, +2)
  i_13 = _get_index_tensor(grid, +0, +2)
  i_23 = _get_index_tensor(grid, +1, +2)
  i_33 = _get_index_tensor(grid, +2, +2)

#  p_00 = input_[i_00[:, :, :, :, 0], i_00[:, :, :, :, 1], i_00[:, :, :, :, 2], i_00[:, :, :, :, 3]]
#  p_10 = input_[i_10[:, :, :, :, 0], i_10[:, :, :, :, 1], i_10[:, :, :, :, 2], i_10[:, :, :, :, 3]]
#  p_20 = input_[i_20[:, :, :, :, 0], i_20[:, :, :, :, 1], i_20[:, :, :, :, 2], i_20[:, :, :, :, 3]]
#  p_30 = input_[i_30[:, :, :, :, 0], i_30[:, :, :, :, 1], i_30[:, :, :, :, 2], i_30[:, :, :, :, 3]]
#
#  p_01 = input_[i_01[:, :, :, :, 0], i_01[:, :, :, :, 1], i_01[:, :, :, :, 2], i_01[:, :, :, :, 3]]
#  p_11 = input_[i_11[:, :, :, :, 0], i_11[:, :, :, :, 1], i_11[:, :, :, :, 2], i_11[:, :, :, :, 3]]
#  p_21 = input_[i_21[:, :, :, :, 0], i_21[:, :, :, :, 1], i_21[:, :, :, :, 2], i_21[:, :, :, :, 3]]
#  p_31 = input_[i_31[:, :, :, :, 0], i_31[:, :, :, :, 1], i_31[:, :, :, :, 2], i_31[:, :, :, :, 3]]
#
#  p_02 = input_[i_02[:, :, :, :, 0], i_02[:, :, :, :, 1], i_02[:, :, :, :, 2], i_02[:, :, :, :, 3]] 
#  p_12 = input_[i_12[:, :, :, :, 0], i_12[:, :, :, :, 1], i_12[:, :, :, :, 2], i_12[:, :, :, :, 3]] 
#  p_22 = input_[i_22[:, :, :, :, 0], i_22[:, :, :, :, 1], i_22[:, :, :, :, 2], i_22[:, :, :, :, 3]] 
#  p_32 = input_[i_32[:, :, :, :, 0], i_32[:, :, :, :, 1], i_32[:, :, :, :, 2], i_32[:, :, :, :, 3]] 
#
#  p_03 = input_[i_03[:, :, :, :, 0], i_03[:, :, :, :, 1], i_03[:, :, :, :, 2], i_03[:, :, :, :, 3]] 
#  p_13 = input_[i_13[:, :, :, :, 0], i_13[:, :, :, :, 1], i_13[:, :, :, :, 2], i_13[:, :, :, :, 3]] 
#  p_23 = input_[i_23[:, :, :, :, 0], i_23[:, :, :, :, 1], i_23[:, :, :, :, 2], i_23[:, :, :, :, 3]] 
#  p_33 = input_[i_33[:, :, :, :, 0], i_33[:, :, :, :, 1], i_33[:, :, :, :, 2], i_33[:, :, :, :, 3]] 

#  p_00 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_10 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_20 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_30 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_01 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_11 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_21 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_31 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_02 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_12 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_22 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_32 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_03 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_13 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_23 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
#  p_33 = Variable(torch.zeros([batch, channel, new_height, new_width]).float().cuda())
  
  p_00 = input_[i_00[:, :, :, :, 0], i_00[:, :, :, :, 1], i_00[:, :, :, :, 2], i_00[:, :, :, :, 3]]
  p_10 = input_[i_10[:, :, :, :, 0], i_10[:, :, :, :, 1], i_10[:, :, :, :, 2], i_10[:, :, :, :, 3]]
  p_20 = input_[i_20[:, :, :, :, 0], i_20[:, :, :, :, 1], i_20[:, :, :, :, 2], i_20[:, :, :, :, 3]]
  p_30 = input_[i_30[:, :, :, :, 0], i_30[:, :, :, :, 1], i_30[:, :, :, :, 2], i_30[:, :, :, :, 3]]

  p_01 = input_[i_01[:, :, :, :, 0], i_01[:, :, :, :, 1], i_01[:, :, :, :, 2], i_01[:, :, :, :, 3]]
  p_11 = input_[i_11[:, :, :, :, 0], i_11[:, :, :, :, 1], i_11[:, :, :, :, 2], i_11[:, :, :, :, 3]]
  p_21 = input_[i_21[:, :, :, :, 0], i_21[:, :, :, :, 1], i_21[:, :, :, :, 2], i_21[:, :, :, :, 3]]
  p_31 = input_[i_31[:, :, :, :, 0], i_31[:, :, :, :, 1], i_31[:, :, :, :, 2], i_31[:, :, :, :, 3]]

  p_02 = input_[i_02[:, :, :, :, 0], i_02[:, :, :, :, 1], i_02[:, :, :, :, 2], i_02[:, :, :, :, 3]] 
  p_12 = input_[i_12[:, :, :, :, 0], i_12[:, :, :, :, 1], i_12[:, :, :, :, 2], i_12[:, :, :, :, 3]] 
  p_22 = input_[i_22[:, :, :, :, 0], i_22[:, :, :, :, 1], i_22[:, :, :, :, 2], i_22[:, :, :, :, 3]] 
  p_32 = input_[i_32[:, :, :, :, 0], i_32[:, :, :, :, 1], i_32[:, :, :, :, 2], i_32[:, :, :, :, 3]] 

  p_03 = input_[i_03[:, :, :, :, 0], i_03[:, :, :, :, 1], i_03[:, :, :, :, 2], i_03[:, :, :, :, 3]] 
  p_13 = input_[i_13[:, :, :, :, 0], i_13[:, :, :, :, 1], i_13[:, :, :, :, 2], i_13[:, :, :, :, 3]] 
  p_23 = input_[i_23[:, :, :, :, 0], i_23[:, :, :, :, 1], i_23[:, :, :, :, 2], i_23[:, :, :, :, 3]] 
  p_33 = input_[i_33[:, :, :, :, 0], i_33[:, :, :, :, 1], i_33[:, :, :, :, 2], i_33[:, :, :, :, 3]]
  
  col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
  col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
  col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
  col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
#  print(p_00.type)
#  print(x_t.type)
  value = _hermite(col0, col1, col2, col3, y_t)
#  print(value.type)
  
  return value
