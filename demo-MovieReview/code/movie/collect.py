from faasit_runtime import FaasitRuntime, function
import json
import time


@function
async def collect_result_handler(frt: FaasitRuntime):
    start_time = time.time()
    params = frt.input()
    login_response = params['input_login']
    review_response = params['input_review']
    recommend_response = params['input_recommend']
    search_response = params['input_search']

    start_input_time = time.time()
    end_input_time = time.time()

    # TODO
    login_message_cnt = len(login_response)
    review_message_cnt = len(review_response)
    recommend_message_cnt = len(recommend_response)
    search_message_cnt = len(search_response)
    end_time = time.time()

    return_val = {
        'process_time': end_time - start_time,
        'input_time': end_input_time - start_input_time,
        'compute_time': 0,
        'output_time': 0,
        'login_message': login_message_cnt,
        'review_message': review_message_cnt,
        'recommend_message': recommend_message_cnt,
        'search_message': search_message_cnt,
    }

    return frt.output({
        **return_val
    })
