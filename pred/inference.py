import datetime
import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(18).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/model_20250315_0407.pt'))
    model.eval()
    # inference
    initial = 999469
    last = initial
    start_time = '2025-03-11 15:03:31'
    for i in range(1, 64):
        hour = i / 4.2342
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, np.log2(initial+1), # time_delta, current_views
                6.319244, 6.96288, 7.04251, 8.38551, 7.648974, 9.061098, 9.147728, 10.07276, 10.653134, 10.092601, 12.008604, 11.676683, 13.230796, # grows_feat
                0.627442, 0.232492, 24.778674 # time_feat
        ]
        np_arr = np.array([data])
        tensor = torch.from_numpy(np_arr).to('cpu', dtype=torch.float32)
        output = model(tensor)
        num = output.detach().numpy()[0][0]
        views_pred = int(np.exp2(num)) + initial
        current_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=hour)
        print(current_time.strftime('%m-%d %H:%M'), views_pred, views_pred - last)
        last = views_pred

if __name__ == '__main__':
    main()