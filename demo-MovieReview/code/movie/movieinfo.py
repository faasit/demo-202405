import json
import time
from faasit_runtime import FaasitRuntime, function

from _hacked import code_dir


@function
async def movie_info_register_handler(frt: FaasitRuntime):
    start_time = time.time()
    params = frt.input()
    user_num = params['user_num']  # 1000000

    start_compute_time = time.time()
    with open((code_dir / 'data.json').as_posix(), 'r', encoding='utf-8') as file:
        file_content = file.read()
        data = json.loads(file_content)

    movie_data = {movie_info["Title"]: movie_info["MovieId"]
                  for movie_info in data}
    user_data = {f'username_{idx}': idx for idx in range(user_num)}
    com_data = {'movie': movie_data, 'user': user_data}
    com_data_recommend = {'movie': data, 'user': user_data}

    end_compute_time = time.time()
    start_output_time = time.time()

    output = dict(
        com_data=com_data,
        com_data_recommend=com_data_recommend,
    )

    end_output_time = time.time()
    end_time = time.time()

    return_val = {
        'process_time': end_time - start_time,
        'input_time': 0,
        'compute_time': end_compute_time - start_compute_time,
        'output_time': end_output_time - start_output_time,
    }

    return frt.output({
        **return_val,
        'output': output
    })
