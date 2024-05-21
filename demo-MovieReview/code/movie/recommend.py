import info as Info
import json
import time
from faasit_runtime import FaasitRuntime, function


@function
async def recommend_handler(frt: FaasitRuntime):
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
    movie_info = [[movie_data['AvgRating'], movie_data['NumRating'], movie_data['MovieId'],
                   movie_data['Title']] for movie_data in movie_info]
    rating_groups = {}

    for movie_data in movie_info:
        rating = int(movie_data[0])
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(movie_data)

    for rating in sorted(rating_groups.keys(), reverse=True):
        movies_in_rating = rating_groups[rating]
        sorted_movies = sorted(
            movies_in_rating, key=lambda x: x[1], reverse=True)
        rating_groups[rating] = sorted_movies

    for single_request in review_request:
        # [req_id, request_type.value, rating, rating_num]
        req_id, request_type, rating, rating_num = single_request
        if request_type != Info.RequestType.Recommend.value:
            response.append([req_id, 'Invalid request type in recommend.'])
            failed_msg += 1
            continue

        movie_list = [
            movie
            for rate in range(rating, 11)
            if 10+rating-rate in rating_groups.keys()
            for movie in rating_groups[10+rating-rate]
            if movie[1] >= rating_num
        ]

        top_k = min(20, len(movie_list))
        movie_list = movie_list[:top_k]
        success_msg += 1

        response.append([req_id, 'Recommend result', movie_list])
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
