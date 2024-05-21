from faasit_runtime import workflow, create_handler,function
from faasit_runtime.workflow import WorkFlowBuilder
from faasit_runtime.runtime import FaasitRuntime
import train_intent_classifier as train
import splitchatbot as split

@function
async def execute(frt: FaasitRuntime):
    _in = frt.input()
    split_params = _in['split']
    train_params = _in['train']
    await frt.call('split', split_params)
    await frt.call('train', train_params)

@workflow
def chatbot(builder: WorkFlowBuilder):
    builder.func("split").set_handler(split.main)
    builder.func('upload_BOW').set_handler(split.upload_BOW)
    builder.func('train').set_handler(train.lambda_handler)
    builder.func('load_bow').set_handler(train.load_bow)
    builder.func('load_indents').set_handler(train.load_indents)
    builder.executor().set_handler(execute)

    return builder.build()

handler = create_handler(chatbot)