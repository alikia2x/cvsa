import datetime
import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(16).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/model_20250315_0504.pt'))
    model.eval()
    # inference
    initial = 999917
    last = initial
    start_time = '2025-03-11 18:43:52'
    for i in range(1, 48):
        hour = i / 30
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, np.log2(initial+1), # time_delta, current_views
                5.231997, 6.473876, 7.063624, 7.026946, 6.9753, 8.599954, 9.448747, 7.236474, 10.881226, 12.128971, 13.351179, # grows_feat
                0.7798611111, 0.2541666667, 24.778674 # time_feat
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