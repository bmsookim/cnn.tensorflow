size = 48
upscaled_size = 48

epochs = 200
batch_size = 128

classifier = 'ED7Classifier_ResNet16'

display_iter = (batch_size*75)
step1 = (60*((57419/batch_size)+1))+1
step2 = (120*((57419/batch_size)+1))+1
step3 = (160*((57419/batch_size)+1))+1