import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(10).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/play_predictor.pth'))
    model.eval()
    # inference
    last = 999469
    for i in range(1, 48):
        hour = i / 2
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, 19.9295936113, # time_delta, current_views
                6.1575520046,8.980,10.6183855023,12.0313328273,13.2537252486, # growth_feat
                0.625,0.2857142857,24.7794093257 # time_feat
        ]
        np_arr = np.array([data])
        tensor = torch.from_numpy(np_arr).to('cpu', dtype=torch.float32)
        output = model(tensor)
        num = output.detach().numpy()[0][0]
        views_pred = int(np.exp2(num)) + 999469
        print(f"{int(15+hour)%24:02d}:{int((15+hour)*60)%60:02d}", views_pred, views_pred - last)
        last = views_pred

if __name__ == '__main__':
    main()