import numpy as np
import torch
from torch.autograd import Variable

"""
  Args :
    srcImage : [batch_size, channel, height, width].
        
  """

def _hermite(A, B, C, D, t):
  a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
  b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
  c = A * (-0.5) + C * 0.5
  d = B

  return a*t*t*t + b*t*t + c*t + d

def GetPixelClamped(image, x, y):

  height = image.shape[0]
  width = image.shape[1]
  y = np.clip(y, 0, height-1)
  x = np.clip(x, 0, width-1)

  return image[y, x]

def SampleBicubic(image, u, v):

  # calculate coordinates -> also need to offset by half a pixel to keep image from shifting down and left half a pixel
  
  height = image.shape[0]
  width = image.shape[1]

  x = (u * width)
  xint = int(x)
  xfract = x - np.floor(x)
 
  y = (v * height)
  yint = int(y)
  yfract = y - np.floor(y)
 
  p00 = GetPixelClamped(image, xint - 1, yint - 1)
  p10 = GetPixelClamped(image, xint + 0, yint - 1)
  p20 = GetPixelClamped(image, xint + 1, yint - 1)
  p30 = GetPixelClamped(image, xint + 2, yint - 1)
 
  p01 = GetPixelClamped(image, xint - 1, yint + 0)
  p11 = GetPixelClamped(image, xint + 0, yint + 0)
  p21 = GetPixelClamped(image, xint + 1, yint + 0)
  p31 = GetPixelClamped(image, xint + 2, yint + 0)
 
  p02 = GetPixelClamped(image, xint - 1, yint + 1)
  p12 = GetPixelClamped(image, xint + 0, yint + 1)
  p22 = GetPixelClamped(image, xint + 1, yint + 1)
  p32 = GetPixelClamped(image, xint + 2, yint + 1)
 
  p03 = GetPixelClamped(image, xint - 1, yint + 2)
  p13 = GetPixelClamped(image, xint + 0, yint + 2)
  p23 = GetPixelClamped(image, xint + 1, yint + 2)
  p33 = GetPixelClamped(image, xint + 2, yint + 2)
 
    # interpolate bi-cubically!
    # Clamp the values since the curve can put the value below 0 or above 255
  col0 = _hermite(p00, p10, p20, p30, xfract)
  col1 = _hermite(p01, p11, p21, p31, xfract)
  col2 = _hermite(p02, p12, p22, p32, xfract)
  col3 = _hermite(p03, p13, p23, p33, xfract)
  value = _hermite(col0, col1, col2, col3, yfract)
  
  return value

 
def ResizeImage(srcImage, scale):
  src_height = srcImage.shape[0]
  src_width = srcImage.shape[1]
  dest_width = int(src_width * scale)
  dest_height = int(src_height * scale)
  destImage = torch.autograd.Variable(torch.zeros([dest_height, dest_width]).float().cuda())
#  print(destImage.shape)
 
  for y in np.linspace(0, dest_height, num = dest_height, endpoint = False, dtype = int):
      v = float(y) / float(dest_height)
      for x in np.linspace(0, dest_width, num = dest_width, endpoint = False, dtype = int):
          u = float(x) / float(dest_width)
          destImage[y, x] = SampleBicubic(srcImage, u, v)

  return destImage
        
    
