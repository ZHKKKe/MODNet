import torch
import torch.nn as nn

from src.models.modnet import MODNet

def makeStateDict(modelPath):
    modnet = MODNet(backbone_pretrained=False)

    torch.save(modnet.state_dict(), modelPath)

def loadStateDict(modelPath):
    modelState = torch.load(modelPath, map_location=torch.device('cpu'))

    state = {}

    prefix = "module."
    for key in modelState:
        stateKey = prefix + key
        state[stateKey] = modelState[key]
    return state

def main():
    modelPath = "models/model.ckpt"
    pretrainedModelPath = "pretrained/modnet_webcam_portrait_matting.ckpt"

    makeStateDict(modelPath)

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    state = loadStateDict(modelPath)
    stateKeys = list(state.keys())
    print(f"state keys {stateKeys[:5]}")

    modnet.load_state_dict(state)

    pretrainedState = torch.load(pretrainedModelPath, map_location=torch.device('cpu'))
    pretrainedStateKeys = list(pretrainedState.keys())
    print(f"pretrainedState keys {pretrainedStateKeys[:5]}")

    modnet.load_state_dict(pretrainedState)

    print(f"state {len(stateKeys)}, preptrainedState {len(pretrainedStateKeys)}, intersection {len(set(stateKeys) & set(pretrainedStateKeys))}")

if __name__ == "__main__":
    main()
