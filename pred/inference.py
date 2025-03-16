import datetime
import numpy as np
from model import CompactPredictor
import torch

def main():
    model = CompactPredictor(16).to('cpu', dtype=torch.float32)
    model.load_state_dict(torch.load('./pred/checkpoints/model_20250315_0530.pt'))
    model.eval()
    # inference
    initial = 99906
    last = initial
    start_time = '2025-03-16 14:48:42'
    for i in range(1, 48):
        hour = i / 4
        sec = hour * 3600
        time_d = np.log2(sec)
        data = [time_d, np.log2(initial+1), # time_delta, current_views
                2.456146, 3.562719, 4.106399, 1.0, 1.0, 5.634413, 6.619818, 1.0, 8.608774, 10.19127, 11.412958, # grows_feat
                0.617153, 0.945308, 22.091431 # time_feat
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