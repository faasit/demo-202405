import info as Info
import json
import time
from faasit_runtime import FaasitRuntime, function


@function
async def search_handler(frt: FaasitRuntime):
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
    movie_info = {movie_data['Title']: movie_data for movie_data in movie_info}
    # rating_groups = {}

    for single_request in review_request:
        # [req_id, request_type.value, search_title]
        req_id, request_type, search_title = single_request
        if request_type != Info.RequestType.Search.value:
            response.append([req_id, 'Invalid request type in search.'])
            failed_msg += 1
            continue

        movie = movie_info.get(search_title)
        if movie is None:
            response.append([req_id, 'No movie is found.'])
            continue

        response_msg = f"Found movie {movie['Title']}(rating: {movie['AvgRating']} by {movie['NumRating']} users)"

        response.append([req_id, response_msg])
        success_msg += 1

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
        'output': output
    })
