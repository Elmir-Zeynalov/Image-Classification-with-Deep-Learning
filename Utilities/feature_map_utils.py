import torch
import torchvision.models as models 
import matplotlib.pyplot as plt
from .utils import load_presaved_model

# storing feature hooks
class FeatureHook:
    def __init__(self, name):
        self.feature_maps = []
        self.batch_indices = []
        self.name = name

    #def hook_fn(self, model, input, output):
    #    self.feature_maps.append(output.clone().detach())
    
    def hook_fn(self, module, input, output):
        self.feature_maps.append(output)
        self.batch_indices.append(module.batchindex)

class FeatureHook:
    def __init__(self, name):
        self.name = name
        self.feature_maps = []
        self.batch_indices = []

    def hook_fn(self, module, input, output):
        self.feature_maps.append(output)
        self.batch_indices.append(module.batchindex)

def feature_map_statistics(test_dataloader, device, epoch, seed):
    model = load_presaved_model(device, seed, epoch-1)
    analyze_feature_maps(model, test_dataloader, device, num_samples=200)

def analyze_feature_maps(model, dataloader, device, num_samples=200):
    # Define ANSI escape code for red color
    RED = '\033[91m'
    # Define ANSI escape code to reset color
    RESET = '\033[0m'

    model.eval()
    print("\n\t\tAnalyzing feature maps!")
    print("*******************************************************************************")
    target_layers = ['conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1', 'avgpool']
    print("Layers - ", target_layers)

    hook_objects = [FeatureHook(name) for name in target_layers]
    samples_analyzed = 0
    for batch_idx, data_batch in enumerate(dataloader):
        data = data_batch['image'].to(device)
        target = data_batch['label'].to(device)

        # Set batch index for each module
        for name, mod in model.named_modules():
            if name in target_layers:
                mod.batchindex = batch_idx

        # Register hooks to target layers in the model
        for name, mod in model.named_modules():
            if name in target_layers:
                mod.register_forward_hook(hook_objects[target_layers.index(name)].hook_fn)

        # Forward pass through the model
        with torch.no_grad():
            output = model(data)
            #print(target)
            #print(output.max(1, keepdim=True)[1][:,0])
            #print()

 
        samples_analyzed += data.size(0)
        if samples_analyzed >= num_samples:
            break

    # Analyze collected feature maps
    average_percentage = []

    for hook_obj in hook_objects:
        for feature_map, batch_index in zip(hook_obj.feature_maps, hook_obj.batch_indices):
            non_positive_percentage = (feature_map <= 0).float().mean().item()
            average_percentage.append(non_positive_percentage)

    if average_percentage:  # Check if the list is not empty
        final_average_percentage = sum(average_percentage) / len(average_percentage)
        #print(average_percentage)
        print(f'->{RED}Average Percentage of Non-Positive Values: {final_average_percentage * 100:.2f}%{RESET}')

        print("*******************************************************************************")
        print()
    else:
        print("No feature maps were collected.")

    for hook_obj in hook_objects:
        hook_obj.feature_maps = []
        hook_obj.batch_indices = []