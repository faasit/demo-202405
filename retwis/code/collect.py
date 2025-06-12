from faasit_runtime import function, FaasitRuntime
import json, time

@function
def CollectResult(frt : FaasitRuntime):
	start_time = time.time()
	params = frt.input()
	ia_login = params['input_login']
	ia_profile = params['input_profile']
	ia_post = params['input_post']
	ia_timeline = params['input_timeline']
	login_response = None
	profile_response = None
	post_response = None
	timeline_response = None
	
	store = frt.storage
	start_input_time = time.time()
	login_response = store.get(ia_login, src_stage='stage2')
	profile_response = store.get(ia_profile, src_stage='stage3')
	post_response = store.get(ia_post, src_stage='stage4')
	timeline_response = store.get(ia_timeline, src_stage='stage5')
	# login_response = md.get_object('stage2', ia_login)
	# profile_response = md.get_object('stage3',ia_profile)
	# post_response = md.get_object('stage4',ia_post)
	# timeline_response = md.get_object('stage5',ia_timeline)
	end_input_time = time.time()

	#TODO
	login_message_cnt = len(login_response)
	profile_message_cnt = len(profile_response)
	post_message_cnt = len(post_response)
	timeline_message_cnt = len(timeline_response)
	end_time = time.time()

	return_val = {
		'process_time': end_time - start_time,
		'input_time': end_input_time - start_input_time,
		'compute_time': 0,
		'output_time': 0,
		'login_message': login_message_cnt,
		'profile_message': profile_message_cnt,
		'post_message': post_message_cnt,
		'timeline_message': timeline_message_cnt,
	}
	
	return {
		'statusCode': 200,
		'body': json.dumps(return_val)
	}

collectResult = CollectResult.export()