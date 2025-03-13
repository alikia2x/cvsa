import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(10).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('play_predictor.pth'))
    model.eval()
    # inference
    data = [3,3.9315974229,5.4263146604,9.4958550269,10.9203528554,11.5835529305,13.0426853722,0.7916666667,0.2857142857,24.7794093257]
    np_arr = np.array([data])
    tensor = torch.from_numpy(np_arr).to('cpu', dtype=torch.float32)
    output = model(tensor)
    print(output)

if __name__ == '__main__':
    main()