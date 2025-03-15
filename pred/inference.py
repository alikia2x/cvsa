import datetime
import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(16).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/model_20250315_0530.pt'))
    model.eval()
    # inference
    initial = 999269
    last = initial
    start_time = '2025-03-15 01:03:21'
    for i in range(1, 48):
        hour = i / 0.5
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, np.log2(initial+1), # time_delta, current_views
                2.801318, 3.455128, 3.903391, 3.995577, 4.641488, 5.75131, 6.723868, 6.105322, 8.141023, 9.576701, 10.665067, # grows_feat
                0.043993, 0.72057, 28.000902 # time_feat
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