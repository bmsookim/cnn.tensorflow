--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
--local ReLU = nn.PReLU --cudnn.ReLU
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function Dropout()
   return nn.Dropout(opt and opt.dropout or 0,nil,true)
end

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end


   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   local blocks = {}

   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(nInputPlane))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(nBottleneckPlane))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end


   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local function wide_layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))
      model:add(layer(block, 512, def[4], 2))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 1000))
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth-4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth-4) / 6
      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      -- The wide-resnet CIFAR-10 model
      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(wide_layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(wide_layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(wide_layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(nStages[4]))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth-4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth-4) / 6
      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      -- The wide-resnet CIFAR-100 model
      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(wide_layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(wide_layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(wide_layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(nStages[4]))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
