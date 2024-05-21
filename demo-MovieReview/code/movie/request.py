import json
import random
import time
import info as Info
from faasit_runtime import FaasitRuntime, function

from _hacked import code_dir


@function
async def request_handler(frt: FaasitRuntime):
    start_time = time.time()
    params = frt.input()
    user_num = params['user_num']  # 1000000
    login_message_cnt = 0
    review_message_cnt = 0
    recommend_message_cnt = 0
    search_message_cnt = 0

    start_compute_time = time.time()
    with open((code_dir / 'data.json').as_posix(), 'r', encoding='utf-8') as file:
        file_content = file.read()
        data = json.loads(file_content)
    title_list = [movie_info['Title'] for movie_info in data]
    title_list_len = len(title_list)

    password_threshold = 100

    login_request = []
    review_request = []
    recommend_request = []
    search_request = []

    for req_id in range(1000000):
        request_type = Info.RequestType(random.randint(1, 4))

        if request_type == Info.RequestType.UserLogin:
            username = random.randint(1, user_num)
            password = username if username > password_threshold else 0
            username = f'username_{username}'
            login_request.append(
                [req_id, request_type.value, username, password])
            login_message_cnt += 1

        elif request_type == Info.RequestType.UpdateReview:
            # req_id, review_id, username(to userid), title(to movieid), rating, text
            review_id = req_id
            username = f'username_{random.randint(0, user_num-1)}'
            title = title_list[random.randint(0, title_list_len-1)]
            rating = random.randint(0, 10)
            text = 'Awesome' if rating > 6 else 'Not bad'
            review_request.append([req_id, request_type.value, review_id,
                                   username, title, rating, text])
            review_message_cnt += 1

        elif request_type == Info.RequestType.Recommend:
            rating = random.randint(0, 10)
            rating_num = random.randint(0, 12000)
            recommend_request.append(
                [req_id, request_type.value, rating, rating_num])
            recommend_message_cnt += 1

        elif request_type == Info.RequestType.Search:
            search_title = title_list[random.randint(0, title_list_len-1)]
            if random.randint(1, 10000) > 9500:
                search_title = 'Error'
            search_request.append([req_id, request_type.value, search_title])
            search_message_cnt += 1

    end_compute_time = time.time()
    start_output_time = time.time()

    output = dict(
        login_request=login_request,
        review_request=review_request,
        recommend_request=recommend_request,
        search_request=search_request,
    )

    end_output_time = time.time()
    end_time = time.time()
    return_val = {
        'process_time': end_time - start_time,
        'input_time': 0,
        'compute_time': end_compute_time - start_compute_time,
        'output_time': end_output_time - start_output_time,
        'login_message': login_message_cnt,
        'review_message': review_message_cnt,
        'recommend_message': recommend_message_cnt,
        'search_message': search_message_cnt,
    }

    return frt.output({
        **return_val,
        'output': output
    })
