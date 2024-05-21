import info as Info
import json
import time
from faasit_runtime import FaasitRuntime, function


@function
async def review_update_handler(frt: FaasitRuntime):
    start_time = time.time()
    params = frt.input()
    info = params['input_info']
    review_request = params['input_request']
    failed_msg, success_msg = 0, 0

    response = []
    start_input_time = time.time()

    end_input_time = time.time()
    start_compute_time = time.time()
    movie_info: dict = info['movie']
    user_info: dict = info['user']

    for single_request in review_request:
        # [req_id, review_id, username(to userid), title(to movieid), rating, text]
        req_id, request_type, review_id, username, title, rating, text = single_request
        if request_type != Info.RequestType.UpdateReview.value:
            response.append([req_id, 'Invalid request type in review update.'])
            failed_msg += 1
            continue
        user_id = user_info.get(username)
        movie_id = movie_info.get(title)

        if user_id is None or movie_id is None:
            response.append(
                [req_id, 'Invalid username or movie name in review update.'])
            continue

        success_msg += 1
        response.append(
            [req_id, f'{username} review movie_{movie_id} success.'])

    end_compute_time = time.time()
    start_output_time = time.time()
    output = dict(
        response=response,
    )

    end_output_time = time.time()
    end_time = time.time()

    return_val = {
        'process_time': end_time - start_time,
        'input_time': end_input_time - start_input_time,
        'compute_time': end_compute_time - start_compute_time,
        'output_time': end_output_time - start_output_time,
        'failed_request': failed_msg,
        'success_request': success_msg,
    }

    return frt.output({
        **return_val,
        'output': output,
    })
