import datetime
import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(10).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/long_term.pt'))
    model.eval()
    # inference
    initial = 997029
    last = initial
    start_time = '2025-03-17 00:13:17'
    for i in range(1, 120):
        hour = i / 0.5
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, np.log2(initial+1), # time_delta, current_views
                6.111542, 8.404707, 10.071566, 11.55888, 12.457823,# grows_feat
                 0.009225, 0.001318, 28.001814# time_feat
        ]
        np_arr = np.array([data])
        tensor = torch.from_numpy(np_arr).to('cpu', dtype=torch.float32)
        output = model(tensor)
        num = output.detach().numpy()[0][0]
        views_pred = int(np.exp2(num)) + initial
        current_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=hour)
        print(current_time.strftime('%m-%d %H:%M:%S'), views_pred, views_pred - last)
        last = views_pred

if __name__ == '__main__':
    main()