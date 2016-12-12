--
--  Copyright (c) 2016, DMIS
--  All rights reserved.
--
--  (Author) Bumsoo Kim, 2016
--  Github : https://github.com/meliketoy/ImageRecognition
--
--  Korea University, Data-Mining Lab
--  Image Recognition Torch Implementation
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'image'
local models = require 'networks/init'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local elapsed_time = 0

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

labels = {'cat', 'dog'}

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.best(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)
model:evaluate()

local function findImages(dir)
   local imagePaths = torch.CharTensor()
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i = 2, #extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   
   while true do
      local line = f:read('*line')
      if not line then break end
      local filename = paths.basename(line)
      local path = dir .. filename
      table.insert(imagePaths, path)
      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   local nImages = #imagePaths

   return imagePaths, nImages
end


testImagePath, nImages = findImages('gen/catdog/test/')
count = 0

for i=1, nImages do
   local timer = torch.Timer()
   test_path = testImagePath[i]
   test_image = image.load(test_path)
   interpolation = 'bicubic'
   size = 224

   local w, h = test_image:size(3), test_image:size(2)

   if(w<=h and w==size) or (h<=w and h==size) then
      test_image = test_image
   end

   if w < h then
      test_image = image.scale(test_image, size, h/w * size, interpolation)
   else
      test_image = image.scale(test_image, w/h * size, size, interpolation)
   end

   local w1 = math.ceil((test_image:size(3) -size)/2)
   local h1 = math.ceil((test_image:size(2) -size)/2)

   test_image = image.crop(test_image, w1, h1, w1+size, h1+size)
   -- print(test_image:size())

   test_image:resize(1, 3, 224, 224)

   result = model:forward(test_image):float()
   --print(result)

   exp = torch.exp(result)
   exp_sum = exp:sum()
   exp = torch.div(exp, exp_sum)
   -- print(exp)

   maxs, indices = torch.max(exp, 2)

   if maxs:sum() > 0.8 then
      count = count + 1
      print('The prediction for '..test_path..' is '..
                                 sys.COLORS.red .. labels[indices:sum()] ..
                                 sys.COLORS.none .. ' by ' .. maxs:sum()
                                 .. ' confidence')
   end
   elapsed_time = elapsed_time + timer:time().real
   timer:reset()
end

print(count)
print(elapsed_time)
