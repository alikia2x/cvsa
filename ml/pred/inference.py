import datetime
import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(15).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/model_20250320_0045.pt'))
    model.eval()
    # inference
    initial = 999704
    last = initial
    start_time = '2025-03-19 22:00:42'
    for i in range(1, 48):
        hour = i / 6
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, np.log2(initial+1), # time_delta, current_views
                4.857981, 6.29067, 6.869476, 6.58392, 6.523051, 8.242355, 8.841574, 10.203909, 11.449314, 12.659556, # grows_feat
                0.916956, 0.416708, 28.003162 # time_feat
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