--
--  (Author) Bumsoo Kim, 2016
--  Github : https://github.com/meliketoy/ImageRecognition
--
--  Korea University, Data-Mining Lab
--  Image Recognition Torch Implementation
--
--  The main code for training and testing each datasets.
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'networks/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.best(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)
print(model)

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
          print('==================================================================')
          print(' * Best model (Top1): ', string.format('%5.2f', testTop1)..'%\n'..
                '              (Top5): ', string.format('%5.2f', testTop5)..'%\n')
          print('=> Saving the best model in '..opt.save)
          print('==================================================================\n')
      else
          print('==================================================================')
          print(' * Best model (Top1): ', string.format('%5.2f', testTop1)..'%\n')
          print('=> Saving the best model in '..opt.save)
          print('==================================================================\n')

      end
   end

   -- Save the model (currently on best model save mode)
   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

if opt.top5_display then
    print(string.format(' * Finished Top1: %6.3f  Top5: %6.3f', bestTop1, bestTop5))
else
    print(string.format(' * Finished Top1: %6.3f', bestTop1))
end
