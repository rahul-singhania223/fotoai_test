import torch
import torch.nn as nn
import torch.nn.functional as F


class DCENet(nn.Module):

	def __init__(self):
		super(DCENet, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x, alpha=1.0):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


		x = x + alpha * r1*(torch.pow(x,2)-x)
		x = x + alpha * r2*(torch.pow(x,2)-x)
		x = x + alpha * r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + alpha * r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + alpha * r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + alpha * r6*(torch.pow(x,2)-x)	
		x = x + alpha * r7*(torch.pow(x,2)-x)
		enhance_image = x + alpha * r8*(torch.pow(x,2)-x)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return enhance_image_1,enhance_image,r


# class DCENet(nn.Module):
#     """DCENet Module"""

#     def __init__(self, n_filters=32):
#         super(DCENet, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=3, out_channels=n_filters,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=n_filters, out_channels=n_filters,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.conv3 = nn.Conv2d(
#             in_channels=n_filters, out_channels=n_filters,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.conv4 = nn.Conv2d(
#             in_channels=n_filters, out_channels=n_filters,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.conv5 = nn.Conv2d(
#             in_channels=n_filters * 2, out_channels=n_filters,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.conv6 = nn.Conv2d(
#             in_channels=n_filters * 2, out_channels=n_filters,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.conv7 = nn.Conv2d(
#             in_channels=n_filters * 2, out_channels=24,
#             kernel_size=3, stride=1, padding=1, bias=True
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, alpha=1.0):
#         x1 = self.relu(self.conv1(x))
#         x2 = self.relu(self.conv2(x1))
#         x3 = self.relu(self.conv3(x2))
#         x4 = self.relu(self.conv4(x3))
#         x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
#         x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
#         x_r = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))
#         r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
#         x = x + alpha * r1 * (torch.pow(x, 2) - x)
#         x = x + alpha * r2 * (torch.pow(x, 2) - x)
#         x = x + alpha * r3 * (torch.pow(x, 2) - x)
#         enhance_image_1 = x + alpha * r4 * (torch.pow(x, 2) - x)
#         x = enhance_image_1 + alpha * r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
#         x = x + alpha * r6 * (torch.pow(x, 2) - x)
#         x = x + alpha * r7 * (torch.pow(x, 2) - x)
#         enhance_image = x + alpha * r8 * (torch.pow(x, 2) - x)
#         r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
#         return enhance_image_1, enhance_image, r


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)