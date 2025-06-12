from faasit_runtime import workflow, Workflow


@workflow
def retwisworkflow(wf: Workflow):
    s0 = wf.call('stage0', {
        "output": "Retwis/stage0/movie_info",
        "max_users": 5000,
        "max_followers": 100,
        "max_posts": 500,
        "post_length": 20
    })
    
    # output: Retwis/stage1/request
    # max_users: 10000
    # post_length: 20
    s1 = wf.call("stage1", {
        "max_users": 10000,
        "post_length": 20,
        "output": "Retwis/stage1/request"
    })

    # input_info: Retwis/stage0/movie_info-2
    # input_request: Retwis/stage1/request-2
    # output: Retwis/stage2/userlogin
    s2 = wf.call("stage2", {
        "input_info": "Retwis/stage0/movie_info-2",
        "input_request": "Retwis/stage1/request-2",
        "output": "Retwis/stage2/userlogin",
        "s0": s0,
        "s1": s1
    })

    # input_info: Retwis/stage0/movie_info-3
    # input_request: Retwis/stage1/request-3
    # output: Retwis/stage3/profile
    s3 = wf.call('stage3', {
        "input_info": "Retwis/stage0/movie_info-3",
        "input_request": "Retwis/stage1/request-3",
        "output": "Retwis/stage3/profile",
        "s0": s0,
        "s1": s1
    })

    # input_info: Retwis/stage0/movie_info-4
    # input_request: Retwis/stage1/request-4
    # update: Retwis/stage4/update
    # newpost: Retwis/stage4/newpost
    # output: Retwis/stage4/post
    s4 = wf.call("stage4", {
        "input_info": "Retwis/stage0/movie_info-4",
        "input_request": "Retwis/stage1/request-4",
        "output": "Retwis/stage4/post",
        "update": "Retwis/stage4/update",
        "newpost": "Retwis/stage4/newpost",
        "s0": s0,
        "s1": s1
    })

    # input_info: Retwis/stage0/movie_info-5
    # input_request: Retwis/stage1/request-5
    # update: Retwis/stage4/update
    # newpost: Retwis/stage4/newpost
    # output: Retwis/stage5/timeline
    s5 = wf.call("stage5", {
        "input_info": "Retwis/stage0/movie_info-5",
        "input_request": "Retwis/stage1/request-5",
        "output": "Retwis/stage5/timeline",
        "update": "Retwis/stage4/update",
        "newpost": "Retwis/stage4/newpost",
        "s0": s0,
        "s1": s1
    })

    # input_login: Retwis/stage2/userlogin
    # input_profile: Retwis/stage3/profile
    # input_post: Retwis/stage4/post
    # input_timeline: Retwis/stage5/timeline
    # output: Retwis/stage6/updatereview
    s6 = wf.call("stage6", {
        "input_login": "Retwis/stage2/userlogin",
        "input_profile": "Retwis/stage3/profile",
        "input_post": "Retwis/stage4/post",
        "input_timeline": "Retwis/stage5/timeline",
        "output": "Retwis/stage6/updatereview",
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5
    })

    return s6

retwisworkflow = retwisworkflow.export()

