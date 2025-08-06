---
layout: post
title: Langchain-Langserve-Langgraph复习
tags: ["Langchain","Langserve","Langgraph"]
categories: ["人工智能"]
---

## Langchain

### PromptTemplate

提示词模板的实例化方式

```python
from langchain_core.prompts import PromptTemplate

# Instantiation using from_template (recommended)
prompt = PromptTemplate.from_template("Say {foo}")
prompt.format(foo="bar")

# Instantiation using initializer
prompt = PromptTemplate(template="Say {foo}")

print(prompt.get_input_schema())
# <class 'langchain_core.utils.pydantic.PromptInput'>

example_prompt = PromptTemplate.from_examples(
    ["hello,world {user}!", "hello,python {user}!"],
    input_variables=["user"],
    example_separator="\n",
    suffix="hello",
    prefix="FUCK"
)

print(example_prompt.format(user="qingchen"))
# FUCK
# hello,world qingchen!
# hello,python qingchen!
# hello

template = PromptTemplate.from_template(
    "hhh{p1}{p2}{p3}",
    partial_variables={"p1": "x", },
)

```

### QingchenModel

自己封装了的model方便切换模型

```python
from langchain_openai import ChatOpenAI
from llm_env import get_env


class QingchenModel(ChatOpenAI):
    def __init__(
            self,
            cache=None,
            temperature=0.9,
            verbose=False,
            streaming=False,
            # api_key=get_env.load_env_ali_api_key(),
            # model='qwen-max-latest',
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model='moonshot-v1-8k',
            api_key=get_env.load_env_moonshot_api_key(),
            base_url="https://api.moonshot.cn/v1",
            **kwargs
    ):
        ChatOpenAI.__init__(
            self,
            verbose=verbose,
            temperature=temperature,
            streaming=streaming,
            # api_key=get_env.load_env_glm_api_key(),
            api_key=api_key,
            model=model,
            base_url=base_url,
            cache=cache,
            **kwargs
        )
```

### RunnableAssign

就是把输入的dict和返回的结果合到一起

```python
# This is a RunnableAssign
from typing import Dict
from langchain_core.runnables.passthrough import (
    RunnableAssign,
    RunnableParallel,
)
from langchain_core.runnables.base import RunnableLambda
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from operator import itemgetter

from langchain_qingchen.chat.QingchenModel import QingchenModel


def add_ten(x: Dict[str, int]) -> Dict[str, int]:
    return {"added": x["input"] + 10}


mapper = RunnableParallel(
    {"add_step": RunnableLambda(add_ten)}
)
print(mapper.invoke({"input": 5}))
# {'add_step': {'added': 15}}

runnable_assign = RunnableAssign(mapper)

# Synchronous example
print(runnable_assign.invoke({"input": 5}))
# returns {'input': 5, 'add_step': {'added': 15}}

# Asynchronous example
# await runnable_assign.ainvoke({"input": 5})
# returns {'input': 5, 'add_step': {'added': 15}}


prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
)
print(prompt)
# llm = FakeStreamingListLLM(responses=["foo-lish"])
llm = QingchenModel()

chain: Runnable = prompt | llm | {"str": StrOutputParser()}

print(chain.invoke({"question": "顺便生成一个笑话"}))

chain_with_assign = chain.assign(hello=itemgetter("str") | llm)
# 这里是这样的输入是{"question": "xxx"} -> {"str": "llm response"} -> （assgin会把输入也带下来）变成{"str": "llm response1","hello":"llm response2"}

# print(chain_with_assign.input_schema.model_json_schema())

# print(chain_with_assign.output_schema.model_json_schema())

response = chain_with_assign.invoke({"question": "顺便生成一个笑话"})
print(response)


# {'add_step': {'added': 15}}
# {'input': 5, 'add_step': {'added': 15}}
# input_variables=['question'] input_types={} partial_variables={} messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a nice assistant.'), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'], input_types={}, partial_variables={}, template='{question}'), additional_kwargs={})]
# {'str': '当然可以，这里有一个幽默的笑话供您欣赏：\n\n有一天，一只狗走进了一个电脑商店，对售货员说：“给我来两台电脑，一台给我，另一台给我的猫。”\n\n售货员惊讶地问：“猫也需要电脑吗？”\n\n狗回答说：“是的，我的猫非常聪明，它可以一边上网冲浪，一边砸掉所有不喜欢它的老鼠。”\n\n希望这个笑话能让您开心一笑！'}
# {'str': '当然可以，这里有一个轻松幽默的笑话：\n\n有一天，一位电脑程序员去找他的医生并抱怨说：“医生，我觉得我得了抑郁症。我真的很不喜欢我的工作。”\n医生回答说：“为什么不尝试换个工作呢？比如，你可以尝试去动物园工作。”\n程序员疑惑地问：“但是我对动物一无所知啊。”\n医生笑着说：“没关系，你可以当一个人类的主键（Primary Key）。”\n\n笑话解释：在这个笑话中，“主键”是数据库术语，指的是一个能够区分数据库中每个记录的唯一标识符。这里医生用了一个双关语，暗示程序员虽然是人，但对动物知之甚少，所以在动物园里就像是一个用来区分不同人类的唯一标识符，即“人类的主键”。', 'hello': AIMessage(content='哈哈，这个笑话确实很有趣，它巧妙地结合了程序员的专业术语和日常生活，让人会心一笑。如果你想了解更多这样的笑话，或者有其他问题需要帮助，随时告诉我！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 39, 'prompt_tokens': 157, 'total_tokens': 196, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'moonshot-v1-8k', 'system_fingerprint': None, 'id': 'chatcmpl-689232111e91f778fae383a1', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--d8b92cf9-9299-40cf-8b94-0c3655d9fd0e-0', usage_metadata={'input_tokens': 157, 'output_tokens': 39, 'total_tokens': 196, 'input_token_details': {}, 'output_token_details': {}})}
```

### RunnableGenerator

接收一个迭代器，然后可以流式返回

```python
# @Author  : ljl
# @Time    : 2025/1/17 下午2:44
import asyncio
from typing import Any, AsyncIterator, Iterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_qingchen.chat.QingchenModel import QingchenModel


def gen(input: Iterator[Any]) -> Iterator[str]:
    for token in ["Have", " a", " nice", " day"]:
        yield token


runnable = RunnableGenerator(gen)
print(runnable.invoke(None))
print(list(runnable.stream(None)))  # ["Have", " a", " nice", " day"]
print(runnable.batch([None, None]))  # ['Have a nice day', 'Have a nice day']


# Async version:
async def agen(input: AsyncIterator[Any]) -> AsyncIterator[str]:
    for token in ["Have", " a", " nice", " day"]:
        yield token


async def main():
    runnable = RunnableGenerator(agen)
    result = await runnable.ainvoke(None)
    print(result)  # "Have a nice day"
    [print(p) async for p in runnable.astream(None)]  # ["Have", " a", " nice", " day"]


# 运行异步主函数
asyncio.run(main())

model = QingchenModel()
chant_chain = (
        ChatPromptTemplate.from_template("Give me a 3 word chant about {topic}")
        | model
        | StrOutputParser()
)


def character_generator(input: Iterator[str]) -> Iterator[str]:
    for token in input:
        if "," in token or "." in token:
            yield "👏" + token
        else:
            yield token


runnable = chant_chain | character_generator
assert type(runnable.last) is RunnableGenerator # 这里会自动封装成RunnableGenerator
print("".join(runnable.stream({"topic": "waste"})))


# Note that RunnableLambda can be used to delay streaming of one step in a
# sequence until the previous step is finished:
# RunnableLambda可用于延迟序列中某一步骤的流式处理，直到前一步骤完成
def reverse_generator(input: str) -> Iterator[str]:
    # Yield characters of input in reverse order.
    for character in input[::-1]:
        yield character


runnable = chant_chain | RunnableLambda(reverse_generator)
print("".join(runnable.stream({"topic": "waste"})))

# Have a nice day
# ['Have', ' a', ' nice', ' day']
# ['Have a nice day', 'Have a nice day']
# Have a nice day
# Have
#  a
#  nice
#  day
# Reduce👏, Reuse👏, Recycle
# elcyceR ,esueR ,ecudeR
```

### RunnableLambda

我平时写用的很多，可以写一个方法用RunnableLambda包一下就能直接用上

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.globals import set_debug
from langchain_core.tracers import ConsoleCallbackHandler
import random

set_debug(True)


def add_one(x: int) -> int:
    return x + 1


def buggy_double(y: int) -> int:
    """Buggy code that will fail 70% of the time"""
    if random.random() > 0.3:
        print('This code failed, and will probably be retried!')  # noqa: T201
        raise ValueError('Triggered buggy code')
    return y * 2


sequence = (
        RunnableLambda(add_one) |
        RunnableLambda(buggy_double).with_retry(  # Retry on failure
            stop_after_attempt=10,
            wait_exponential_jitter=False
        )
)

print(sequence.input_schema.model_json_schema())  # Show inferred input schema
print(sequence.output_schema.model_json_schema())  # Show inferred output schema
# print(sequence.invoke(2, config={'callbacks': [ConsoleCallbackHandler()]}))  # invoke the sequence (note the retry above!!)
print(sequence.invoke(2))  # invoke the sequence (note the retry above!!)

# {'title': 'add_one_input', 'type': 'integer'}
# {'title': 'buggy_double_output', 'type': 'integer'}
# [chain/start] [chain:RunnableSequence] Entering Chain run with input:
# {
#   "input": 2
# }
# [chain/start] [chain:RunnableSequence > chain:add_one] Entering Chain run with input:
# {
#   "input": 2
# }
# [chain/end] [chain:RunnableSequence > chain:add_one] [0ms] Exiting Chain run with output:
# {
#   "output": 3
# }
# [chain/start] [chain:RunnableSequence > chain:buggy_double] Entering Chain run with input:
# {
#   "input": 3
# }
# [chain/start] [chain:RunnableSequence > chain:buggy_double > chain:buggy_double] Entering Chain run with input:
# {
#   "input": 3
# }
# [chain/end] [chain:RunnableSequence > chain:buggy_double > chain:buggy_double] [0ms] Exiting Chain run with output:
# {
#   "output": 6
# }
# [chain/end] [chain:RunnableSequence > chain:buggy_double] [1ms] Exiting Chain run with output:
# {
#   "output": 6
# }
# [chain/end] [chain:RunnableSequence] [2ms] Exiting Chain run with output:
# {
#   "output": 6
# }
# 6

```

### RunnableParallel

并行

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

from langchain_qingchen.chat.QingchenModel import QingchenModel


def add_one(x: int) -> int:
    return x + 1


def mul_two(x: int) -> int:
    return x * 2


def mul_three(x: int) -> int:
    return x * 3


runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)

sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
    "mul_two": runnable_2,
    "mul_three": runnable_3,
}
# Or equivalently:
# sequence = runnable_1 | RunnableParallel(
#     {"mul_two": runnable_2, "mul_three": runnable_3}
# )
# Also equivalently:
# sequence = runnable_1 | RunnableParallel(
#     mul_two=runnable_2,
#     mul_three=runnable_3,
# )

print(sequence.invoke(1))
# await sequence.ainvoke(1)

print(sequence.batch([1, 2, 3]))
# await sequence.abatch([1, 2, 3])


model = QingchenModel()
joke_chain = (
        ChatPromptTemplate.from_template("给我讲个关于{topic}的笑话")
        | model
)
poem_chain = (
        ChatPromptTemplate.from_template("写一首关于{topic}的两行诗")
        | model
)

runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

# Display stream
# output = {key: "" for key, _ in runnable.output_schema()}
for chunk in runnable.stream({"topic": "泳衣"}):
    for key in chunk:
        print(chunk)  # noqa: T201

# {'mul_two': 4, 'mul_three': 6}
# [{'mul_two': 4, 'mul_three': 6}, {'mul_two': 6, 'mul_three': 9}, {'mul_two': 8, 'mul_three': 12}]
# {'joke': AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='好的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='，', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='这里', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='有一个', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='关于', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='轻松', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='幽默', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='笑话', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='：\n\n', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='有一天', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='，', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='一位', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='顾客', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='走进', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='店', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='对', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='店员', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='说', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='：“', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='我想', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='找', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='一套', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='既', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='时尚', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='又', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='实用的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='。', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='”\n', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='店员', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='回答', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='：“', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='当然', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='，', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='我们', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='这里有', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='各种各样的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='。', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='”\n', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='顾客', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='说', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='：“', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='我想要', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='一款', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='即使', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='在水中', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='也不会', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='变得', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='透明的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='。', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='”\n', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='店员', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='想了想', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='，', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='然后', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='微笑着', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='说', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='：“', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='那', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='您', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='可能', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='需要', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='一款', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='潜水', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='服', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='。”\n\n', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='这个', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='笑话', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='以', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='的材料', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='和', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='功能', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='为', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='切入点', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='，', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='用', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='一个', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='幽默', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='的', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='转折', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='来', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='达到', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='幽默', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='效果', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='。', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='希望', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='这个', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='笑话', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='能', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='让您', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='会', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='心', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='一笑', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='！', additional_kwargs={}, response_metadata={}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'joke': AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'moonshot-v1-8k', 'system_fingerprint': 'fpv0_ff52a3ef'}, id='run--b900033d-941d-426c-8359-fda5f825cbf9')}
# {'poem': AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='水中', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='舞', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='者', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='披', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='轻', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='纱', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='，\n', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='碧', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='波', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='映', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='日', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='泳', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='衣', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='华', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='。', additional_kwargs={}, response_metadata={}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
# {'poem': AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'moonshot-v1-8k', 'system_fingerprint': 'fpv0_ff52a3ef'}, id='run--a998b997-aca5-4de9-9281-dd63d96d89ad')}
```

### RunnablePassthrough

这个用的也蛮多的

经常这么写：

```python
RunnablePassthrough.assign(test=lambda x: len(x))
# test后面是个runnable的
# RunnablePassthrough会把前面字典带进来,assign把结果加进去，也就是说可以把第一步的模型返回带到第二步最终一起返回
```

```python
import time

from langchain.globals import set_debug
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_qingchen.chat.QingchenModel import QingchenModel


# set_debug(True)
runnable = RunnableParallel(
    origin=RunnablePassthrough(),
    modified=lambda x: x + 1
)

print(runnable.invoke(1))


def fake_llm(prompt: str) -> str:  # Fake LLM for the example
    return "completion"


chain = RunnableLambda(fake_llm) | {
    'original': RunnablePassthrough(),  # Original LLM output
    'parsed': lambda text: text[::-1]  # Parsing logic
}

print(chain.invoke('hello'))

runnable = {
               'llm1': fake_llm,
               'llm2': fake_llm,
           } | RunnablePassthrough.assign(
    total_chars=lambda inputs: len(inputs['llm1'] + inputs['llm2'])
)

print(runnable.invoke('hello'))
# {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}


# def time_wait(input):
#     print("sleeping")
#     time.sleep(2)
#
#
# def print_res(inputs):
#     print(inputs)
#
#
# chain2 = QingchenModel() | RunnablePassthrough(time_wait) | RunnablePassthrough(print_res)
# # print(chain2.invoke('你好'))
#
# messages = [
#     ("human", "开始"),
#     ("ai", "Hello, I am Bulbasaur. What's your name?"),
#     ("human", "Hello. My name is Pikachu."),
#     ("ai", "Nice to meet you, Pikachu."),
#     ("human", "What are you doing?"),
#     ("ai", "I'm exploring this area. What about you?"),
#     ("human", "I'm learning English")
# ]
# # print(chain2.invoke(messages))
# for e in chain2.stream(messages):
#     print(e)

# {'origin': 1, 'modified': 2}
# {'original': 'completion', 'parsed': 'noitelpmoc'}
# {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}

```

### RunnableWithMessageHistory

带对话记忆历史通过session区分不同的人

```python
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    ConfigurableFieldSpec,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_qingchen.chat.QingchenModel import QingchenModel


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


history = get_by_session_id("1")
history.add_message(AIMessage(content="hello"))
print(store)  # noqa: T201

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | QingchenModel()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"session_id": "foo"}}
))

# Uses the store defined in the example above.
print(store)  # noqa: T201

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foo"}}
))

print(store)  # noqa: T201


store = {}


def get_session_history(
        user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]


prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | QingchenModel()

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

with_message_history.invoke(
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
```

### FakeModel

##### 待补充

伪造一个大模型返回来做测试

```python
def fake_model(input) -> Iterator[ChatGenerationChunk]:
    print("into fake_model")
    for c in content:
        yield AIMessageChunk(c)
        
full_chain = ((RunnableLambda(fake_model) | JsonOutputParser()).with_retry(stop_after_attempt=3)
              | RunnablePassthrough.assign(
            test_summary=itemgetter("input") | RunnableLambda(lambda x: x)
        ))
```

### CustomJsonOutputParser

`JsonOutputParser`实际使用中有一个问题如果`json`格式有问题直接就返回`none`了

我们可以继承然后改一下

```python
import random
from json import JSONDecodeError
from typing import Iterator, Optional, Callable, Any
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, Generation
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.json import parse_json_markdown

content1 = """{\n    "target": "进行简单的英语对话交流",\n    "roles": [\n        "Bulbasaur",\n        "Pikachu"\n    ],\n    "link": [\n        "环节1:Bulbasaur：‘Hello, I am Bulbasaur. What\'s your name?’  Pikachu：‘Hello. My name is Pikachu.’",\n        "环节2:Bulbasaur：‘Nice to meet you, Pikachu.’  Pikachu：‘Nice to meet you too.’",\n        "环节3:Bulbasaur：‘I\'m exploring this area. What about you?’  Pikachu：‘What are you doing?’",\n        "结束环节:Bulbasaur：‘Well, see you next time! <结束>’  Pikachu：‘bye’"\n    ],\n    "scene": "In a lush forest, Bulbasaur and Pikachu meet. Bulbasaur is looking around curiously when he sees Pikachu and says:",\n    "actor_line": "Well, see you next time!",\n    "prompt": "Bulbasaur in a forest, looking a bit disappointed, high quality"\n} """

content2 = """how are you?你好吗？"""


def fake_model(input) -> Iterator[ChatGenerationChunk]:
    print("into fake_model")
    # 随机选择一个content
    content = content1 if random.random() > 0.5 else content2
    for c in content:
        yield AIMessageChunk(c)


test_chain = fake_model | JsonOutputParser()
# print(test_chain.invoke("hello"))


# 自定义输出解析器
class CustomJsonOutputParser(JsonOutputParser):
    """
    自定义JSON输出解析器，支持JSON解析和自定义回退逻辑
    """

    def __init__(
            self,
            pydantic_object: Optional[Any] = None,
            fallback_func: Optional[Callable[[str], dict]] = None
    ):
        """
        初始化自定义JSON输出解析器

        :param pydantic_object: 可选的Pydantic模型
        :param fallback_func: 当无法解析JSON时的自定义处理函数
        """
        # 使用父类构造函数
        super().__init__(pydantic_object=pydantic_object)
        # 存储回退函数
        self._fallback_func = fallback_func

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                # 不是json格式返回时
                return self._fallback_func(text)
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                # msg = f"Invalid json output: {text}"
                return self._fallback_func(text)


# 自定义回退函数示例
def custom_fallback(text: str) -> dict:
    """
    自定义回退处理函数

    :param text: 原始文本
    :return: 处理后的字典
    """
    return {
        "original_text": text,
        "processed": text.upper(),
        "length": len(text),
        "parse_status": "custom_fallback"
    }


# 创建解析器实例
parser = CustomJsonOutputParser(fallback_func=custom_fallback)
chain = RunnableLambda(fake_model) | parser

# for e in chain.stream("用json格式输出：{'a':1}"):
#     # print(type(e))
#     print(e)

print(chain.invoke("用json格式输出：{'a':1}"))
```

### 待补充

## Langserve

通过fastapi加上langserve可以快速搭建工作流

```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes, RemoteRunnable
from pydantic import BaseModel

from langchain_qingchen.chat.QingchenModel import QingchenModel


class InputClass(BaseModel):
    userinfo: dict


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    QingchenModel(),
    path="/qingchen",
)

add_routes(
    app,
    RemoteRunnable("http://192.168.1.9:8000/joke/").with_types(input_type=InputClass),
    path="/new/joke"
)

model = QingchenModel()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

### 待补充

## Langgraph

### 快速搭建

```python
# @Author  : ljl
# @Time    : 2025/1/25 下午3:30
# Import relevant functionality
from datetime import datetime

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain_qingchen.chat.QingchenModel import QingchenModel

# Create the agent
memory = MemorySaver()
model = QingchenModel()


@tool
def get_time():
    """
    获取当前时间
    :return: 当前时间
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_location():
    """
    获取用户当前位置
    :return: 用户当前位置
    """
    return '南京市浦口区'


tools = [get_time, get_location]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="我是清尘，我现在在哪？")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="南京现在几点了")]}, config
):
    print(chunk)
    print("----")


# {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '01987b44c25a51af954a59a546322a55', 'function': {'arguments': '{}', 'name': 'get_location'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 131, 'total_tokens': 133, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-V3', 'system_fingerprint': '', 'id': '01987b44bcbf607a03000fd17ff8b722', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--1e85b1da-2476-4034-9258-c05af6e8e796-0', tool_calls=[{'name': 'get_location', 'args': {}, 'id': '01987b44c25a51af954a59a546322a55', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 2, 'total_tokens': 133, 'input_token_details': {}, 'output_token_details': {}})]}}
# ----
# {'tools': {'messages': [ToolMessage(content='南京市浦口区', name='get_location', id='499460a1-383e-4df2-a6e8-33a73af37bc9', tool_call_id='01987b44c25a51af954a59a546322a55')]}}
# ----
# {'agent': {'messages': [AIMessage(content='清尘，你现在的位置是在南京市浦口区。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 156, 'total_tokens': 167, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-V3', 'system_fingerprint': '', 'id': '01987b44c4766cd3fe401958ff480d4b', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--74a43175-bca4-4b10-81f3-14d9fdf6b2dd-0', usage_metadata={'input_tokens': 156, 'output_tokens': 11, 'total_tokens': 167, 'input_token_details': {}, 'output_token_details': {}})]}}
# ----
# {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '01987b44d9c94236ea31c9ab626358b5', 'function': {'arguments': '{}', 'name': 'get_time'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 174, 'total_tokens': 176, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-V3', 'system_fingerprint': '', 'id': '01987b44cfca41732375fcabbdd2e0ff', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--5a140a15-728c-4c7d-a576-1546c126f236-0', tool_calls=[{'name': 'get_time', 'args': {}, 'id': '01987b44d9c94236ea31c9ab626358b5', 'type': 'tool_call'}], usage_metadata={'input_tokens': 174, 'output_tokens': 2, 'total_tokens': 176, 'input_token_details': {}, 'output_token_details': {}})]}}
# ----
# {'tools': {'messages': [ToolMessage(content='2025-08-06 01:26:07', name='get_time', id='a4b538c2-780c-401c-89e6-2c255e5ac4f8', tool_call_id='01987b44d9c94236ea31c9ab626358b5')]}}
# ----
# {'agent': {'messages': [AIMessage(content='南京现在是2025年8月6日凌晨1点26分。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 207, 'total_tokens': 221, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-V3', 'system_fingerprint': '', 'id': '01987b44dc553d3469e67f211c2c2e9d', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--5a662040-3c10-491f-bd04-a3c203f38b25-0', usage_metadata={'input_tokens': 207, 'output_tokens': 14, 'total_tokens': 221, 'input_token_details': {}, 'output_token_details': {}})]}}
# ----
```

### 分步搭建

#### 工具

主要是用`convert_to_openai_tool`转成对应的格式

```python
# @Time    : 2025/1/25 下午2:12
from datetime import datetime
from langchain_core.tools import tool
from typing import List
from langchain_core.utils.function_calling import convert_to_openai_tool
from utils.bingsearch import run_bingsearch_workflow


# 20250325 优化图片搜索工具
@tool
def image_bing_search(keyword: str, count: int, offset: int) -> List[str]:
    """
    使用bing搜图工具
    :param keyword: 搜图提示词
    :param count: 搜索结果数量
    :param offset: 搜索结果偏移量，用于翻页，默认为0-30的随机整数
    :return: 搜图结果是图片链接列表
    """
    import random
    if not offset:
        offset = random.randint(0, 30)
    return run_bingsearch_workflow(keyword, count, offset)


@tool
def current_time() -> str:
    """
    获取当前时间
    :return: 当前时间
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


image_bing_search_tool = convert_to_openai_tool(image_bing_search)
current_time_tool = convert_to_openai_tool(current_time)

tools = [image_bing_search, current_time]
TOOLS = [image_bing_search_tool, current_time_tool]
```

#### 模型

关键在于`bind`绑定工具

```python
# @Time    : 2025/2/8 上午10:02
from assistant.tools import TOOLS
from langchain_qingchen.chat.QingchenModel import QingchenModel

llm_with_tools = QingchenModel(tags=["show", "tools", "agent"]).bind(tools=TOOLS)
```

#### 构建图

```python
# @Author  : ljl
# @Time    : 2025/1/25 下午1:48
# 正常对话包含历史记录缓存，包含一个搜图的工具
# langgraph实现
from typing import Annotated
from langchain_core.messages import BaseMessage, ToolMessage, AIMessageChunk, SystemMessage
from langgraph.constants import END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from assistant.assistant_constants import *
from assistant.model_with_tools import llm_with_tools
from assistant.quick_instructions import quick_instructions_pic, quick_instructions_exercise, quick_instructions_advice, quick_instructions_lesson
from assistant.tools import tools
from utils.mylog import logger


# 入参
class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_uuid: Annotated[str, "临时对话线程UUID"]
    quick_instructions: Annotated[bool, "是否是快捷指令"]


# 入口
def entry(state: AssistantState):
    logger.info(f"AI Assistant Entry: {state}")


# 模型
def chatbot(state: AssistantState):
    dialogue: list[BaseMessage] = state["messages"]
    # 添加系统提示词
    dialogue.insert(0, SystemMessage(content=system_prompt))
    return {"messages": [llm_with_tools.invoke(dialogue)]}


def get_last_message(state: AssistantState) -> BaseMessage:
    if messages := state.get("messages", []):
        message: BaseMessage = messages[-1]
    else:
        raise ValueError(f"没有收到用户消息: {state}")
    return message


# 快捷指令
def quick_instructions(state: AssistantState):
    message = get_last_message(state)
    logger.info(f"AI Assistant Quick Instructions: {message}")
    lesson_content = state["lesson_content"]
    if message.content == Q1: 
        return {"messages": [quick_instructions_pic(lesson_content)], "quick_instructions": False}
    elif message.content == Q2: 
        return {"messages": [quick_instructions_exercise(lesson_content)], "quick_instructions": False}
    elif message.content == Q3: 
        return {"messages": [quick_instructions_advice(lesson_content)], "quick_instructions": False}
    elif Q4 in message.content: 
        return {"messages": [quick_instructions_lesson(lesson_content)], "quick_instructions": False}
    else:
        return {"messages": [AIMessageChunk(content="触发了快捷指令，但未找到对应的指令")], "quick_instructions": False}


# 入口路由
def entry_route(state: AssistantState):
    """
    快捷指令判断路由，双重校验
    """
    message = get_last_message(state)

    # 第一层校验
    if state.get("quick_instructions", False):
        # 第二层校验
        if trigger_word in message.content:
            # 处理快捷指令
            return "quick"
    return "chatbot"


# 图
def build_graph():
    # tool
    tool_node = ToolNode(tools=tools)

    # memory
    memory = MemorySaver()

    # build graph
    graph_builder = StateGraph(AssistantState)

    graph_builder.add_node("entry", entry)
    graph_builder.add_node("quick", quick_instructions)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "entry",
        entry_route,
        {"chatbot": "chatbot", "quick": "quick"},
    )

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    graph_builder.add_edge(START, "entry")
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("quick", END)

    agent_graph = graph_builder.compile(checkpointer=memory)
    return agent_graph


graph = build_graph().with_config({"run_name": "ai_assistant"})


# 流式
async def assistant_chat_stream(inputs: AssistantState):
    config = {"configurable": {"thread_id": inputs.get("thread_uuid")}}
    async for message, metadata in graph.astream(inputs, config, stream_mode="messages"):
        message: BaseMessage = message
        if "show" not in metadata.get("tags", []):
            # logger.info(f"AI Assistant Hide Response: {message}")
            continue
        if isinstance(message, ToolMessage):
            logger.info(f"AI Assistant Tool Message: {message}")
            continue
        elif isinstance(message, AIMessageChunk):
            if message.tool_calls:
                logger.info(f"AI Assistant Tool Calls: {message}")
                continue
        yield message

# 展示graph结构
# def show_graph():
#     from PIL import Image
#     from io import BytesIO
#     import matplotlib.pyplot as plt
#     from langchain_core.runnables.graph import MermaidDrawMethod
#
#     def show_img(path) -> None:
#         img = Image.open(path)
#         plt.axis('off')  # 不显示坐标轴
#         plt.imshow(img)  # 将数据显示为图像，即在二维常规光栅上。
#         plt.show()  # 显示图片
#
#     show_img(BytesIO(graph.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)))
#
#
# show_graph()
```

可以结合langserve快速起一个api

```python
add_routes(
    assistant_router,
    RunnableLambda(assistant_chat_stream).with_types(input_type=AssistantState),
    path="/ai/assistant",
)
```

### 待补充
