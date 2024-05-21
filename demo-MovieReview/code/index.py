import _hacked  # noqa

from faasit_runtime import function, workflow, create_handler
from faasit_runtime.workflow import WorkFlowBuilder
from faasit_runtime import FaasitRuntime

from movie.movieinfo import movie_info_register_handler
from movie.request import request_handler
from movie.userlogin import user_login_handler
from movie.reviewupdate import review_update_handler
from movie.recommend import recommend_handler
from movie.search import search_handler
from movie.collect import collect_result_handler


@function
async def executor(frt: FaasitRuntime):
    request_id = "000000"
    common_args = {'request_id': request_id}

    print("movie_info_register going...")
    movie_info_res = await frt.call('movie_info_register', {
        'user_num': 1000000,
        **common_args,
    })

    print("request going...")
    request_res = await frt.call('request', {
        'user_num': 1000000,
        **common_args,
    })

    print("user_login going...")
    user_login_res = await frt.call('user_login', {
        'input_info': movie_info_res['output']['com_data'],
        'input_request': request_res['output']['login_request'],
        **common_args,
    })

    print("review_update going...")
    review_update_res = await frt.call('review_update', {
        'input_info': movie_info_res['output']['com_data'],
        'input_request': request_res['output']['review_request'],
        **common_args,
    })

    print("recommend going...")
    recommend_res = await frt.call('recommend', {
        'input_info': movie_info_res['output']['com_data_recommend'],
        'input_request': request_res['output']['recommend_request'],
        **common_args,
    })

    print("search going...")
    search_res = await frt.call('search', {
        'input_info': movie_info_res['output']['com_data_recommend'],
        'input_request': request_res['output']['search_request'],
        **common_args,
    })

    print("collect_result going...")
    collect_result_res = await frt.call('collect_result', {
        'input_login': user_login_res['output']['response'],
        'input_review': review_update_res['output']['response'],
        'input_recommend': recommend_res['output']['response'],
        'input_search': search_res['output']['response'],
        **common_args,
    })

    return frt.output({
        'collect_result': collect_result_res,
        'status': 'ok'
    })


@workflow
def create_workflow(builder: WorkFlowBuilder):
    builder.func('movie_info_register').set_handler(
        movie_info_register_handler)
    builder.func('request').set_handler(request_handler)
    builder.func('user_login').set_handler(user_login_handler)
    builder.func('review_update').set_handler(review_update_handler)
    builder.func('recommend').set_handler(recommend_handler)
    builder.func('search').set_handler(search_handler)
    builder.func('collect_result').set_handler(collect_result_handler)
    builder.executor().set_handler(executor)

    return builder.build()


handler = create_handler(create_workflow)
