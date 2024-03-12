import torch
import torchvision.models as models 
import matplotlib.pyplot as plt

# storing feature hooks
class FeatureHook:
    def __init__(self, name):
        self.feature_maps = []
        self.name = name

    def hook_fn(self, model, input, output):
        self.feature_maps.append(output.clone().detach())



def analyze_feature_maps(model, dataloader, device, num_samples=200):
    model.eval()
    print("Analyzing feature maps!")
    #target_layers = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1' , 'layer3.0.conv1', 'layer4']
    
    ##BEST SO FAR 50.14
    target_layers = ['conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1', 'avgpool']
    
    # 64.88%
    #target_layers = ['conv1', 'layer1.0.conv1', 'layer2.1.conv1', 'layer3.2.conv1', 'layer4.1.conv1']

    # 61.13%
    #target_layers = ['conv1', 'layer1.2.conv1', 'layer2.3.conv1', 'layer3.4.conv1', 'layer4.0.conv1']

    # 66.65 %
    #target_layers = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1']

    #target_layers = ['conv1', 'layer2.1.conv1', 'layer3.3.conv1', 'layer4.2.conv1', 'fc']
    #target_layers = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1']

    hook_objects = [FeatureHook(name) for name in target_layers]


    for hook_id, (name, mod) in enumerate(model.named_modules()):
        if name in target_layers:
            mod.register_forward_hook(hook_objects[target_layers.index(name)].hook_fn)


    samples_analyzed = 0

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            data = data_batch['image'].to('cuda') # send data to GPU
            target = data_batch['label'].to('cuda') # send data to GPU

            output = model(data)

            samples_analyzed += data.size(0)
            if samples_analyzed >= num_samples:
                break

    average_percentage = []


    for hook_obj in hook_objects:
        for feature_map in hook_obj.feature_maps:
            non_positive_percentage = (feature_map <= 0).float().mean().item()
            average_percentage.append(non_positive_percentage)

    final_average_percentage = sum(average_percentage) / len(average_percentage)
    print(average_percentage)
    print(f'Average Percentage of Non-Positive Values: {final_average_percentage * 100:.2f}%')

    for hook_obj in hook_objects:
        hook_obj.feature_maps = []








    