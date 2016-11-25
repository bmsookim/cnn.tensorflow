--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   if opt.top5_display then 
       print(string.format(' * Results Top1: %6.3f  Top5: %6.2f%s', top1Err, top5Err, '%'))
   else
       print(string.format(' * Results Top1: %6.2f%s', top1Err, '%'))
   end
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = 0
local bestTop5 = 0
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 > bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      if opt.top5_display then
          print(sys.COLORS.red .. ' * Best model (Top1): ',
                sys.COLORS.magenta .. string.format('%5.2f',testTop1)..
                sys.COLORS.magenta .. '%' ..
                sys.COLORS.red .. ' (Top5): ',
                sys.COLORS.magenta .. string.format('%5.2f', testTop5)..
                sys.COLORS.magenta .. '%\n' .. sys.COLORS.none)
      else
          print(sys.COLORS.red .. ' * Best model (Top1): ',
                sys.COLORS.magenta .. string.format('%5.2f', testTop1)..
                sys.COLORS.magenta .. '%\n' .. sys.COLORS.none)
      end
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

if opt.top5_display then
    print(string.format(' * Finished Top1: %6.3f  Top5: %6.3f', bestTop1, bestTop5))
else
    print(string.format(' * Finished Top1: %6.3f', bestTop1))
end
