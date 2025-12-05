---
layout: post
title: Langgraph Command 的妙用
tags: [LangGraph]
categories: ["Python"]
---

## 场景

做了一个支付的 Agent ：

主要功能是用户选产品、下单、付款、交付的一系列功能。



环节：

用户与 Agent 交互后创建订单环节，

通过 interrupt 返回订单信息，等待用户支付，支付消息(data)通过回调返回，agent 在此暂停等待回调数据；

用户输入了信息，继续正常对话；

收到回调则去交付相关产品信息



问题在于：

用户继续聊天后支付了，回调需要走到支付信息对应的 interrupt 的节点，去执行对应的流程



## 解决方法

我一开始想到是的 checkpoint 恢复，后来发现恢复就是再跑一次，传参进去压根没啥影响



command 我一直用，我以为必须要先触发 interrupt 才能去 resume

后面我直接尝试都传，嘿，您瞧怎么着，真行嘿



关键代码：

```python
Command(
  goto="interrupt_node",  # 直接去对应的节点
  resume={"data": "我付完款了，哥们，真不赖!"}  # 带上 resume 这样就不会 interrupt 了， 会按照顺序执行了
)
```



直接看代码吧！

## 代码

```python
from pprint import pprint

from langchain_core.language_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command


class State(MessagesState):
    interrupt: int
    chat: int


def mock_llm(state: State):
    return {"messages": [{"role": "ai", "content": "下单成功，请支付..."}]}


def interrupt_node(state: State):
    payload = interrupt("订单创建完成，等待哥们你的支付...")
    data = payload.get("data")  # 支付回调
    message = payload.get("message")  # 用户继续沟通

    if data:
        return Command(
            goto="data_node",
            update={
                "interrupt": state.get("interrupt", 0) + 1,
                "messages": [{"role": "ai", "content": data}]
            }
        )

    if message:
        return {
            "chat": state.get("chat", 0) + 1,
            "messages": [
                {"role": "human", "content": message},
                {"role": "ai", "content": "包有的，哥们！"}
            ]
        }

    # Error
    return {"messages": [{"role": "human", "content": "不是哥们，传参呢？胡闹！"}, ]}


def data_node(state: State):
    fake_message = FakeListChatModel(responses=["哥们已经在 🫵 给你发货了"]).invoke("")
    return {"messages": [fake_message]}


graph = StateGraph(State)

graph.add_node(mock_llm)
graph.add_node(interrupt_node)
graph.add_node(data_node)

graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", "interrupt_node")
graph.add_edge("interrupt_node", END)
graph.add_edge("data_node", END)

checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

for model, value in graph.stream(
        {"messages": [{"role": "user", "content": "哥们，我来买一张方大同的黑胶唱片"}]},
        stream_mode=["updates", "checkpoints"],
        config=config
):
    print("*" * 100)
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)
    print()

print("=" * 100)
print("state: ", graph.get_state(config=config))
print()

# 恢复中断
for model, value in graph.stream(
        Command(resume={"message": "你真有啊？"}),
        stream_mode=["updates", "checkpoints"],
        config=config
):
    print("*" * 100)
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)
    print()

print("=" * 100)
print("state: ", graph.get_state(config=config))
print()

for model, value in graph.stream(
        Command(
            goto="interrupt_node",
            resume={"data": "我付完款了，哥们，真不赖!"}
        ),
        stream_mode=["updates", "checkpoints"],
        config=config
):
    print("*" * 100)
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)
    print()

```

输出：

```text
****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2b-68ee-bfff-e57d121d9c93'}},
 'parent_config': None,
 'values': {'messages': []},
 'metadata': {'source': 'input',
              'step': -1,
              'parents': {}},
 'next': ['__start__'],
 'tasks': [{'id': '03ce38ff-7777-e58f-3803-d697e595fe8d',
            'name': '__start__',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c2b-68ee-bfff-e57d121d9c93'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5')]},
 'metadata': {'source': 'loop',
              'step': 0,
              'parents': {}},
 'next': ['mock_llm'],
 'tasks': [{'id': '7e2a12db-50d7-7328-4344-7cef5ad24630',
            'name': 'mock_llm',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'mock_llm': {'messages': [{'role': 'ai',
                            'content': '下单成功，请支付...'}]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b')]},
 'metadata': {'source': 'loop',
              'step': 1,
              'parents': {}},
 'next': ['interrupt_node'],
 'tasks': [{'id': '2ff3e45a-a474-fb08-a15d-631aac7ea187',
            'name': 'interrupt_node',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'__interrupt__': (Interrupt(value='订单创建完成，等待哥们你的支付...',
                             id='b63ac8c675718737076485b99fa18474'),)}

====================================================================================================
state:  StateSnapshot(values={'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'), AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b')]}, next=('interrupt_node',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-11-25T11:25:58.337438+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}}, tasks=(PregelTask(id='2ff3e45a-a474-fb08-a15d-631aac7ea187', name='interrupt_node', path=('__pregel_pull', 'interrupt_node'), error=None, interrupts=(Interrupt(value='订单创建完成，等待哥们你的支付...', id='b63ac8c675718737076485b99fa18474'),), state=None, result=None),), interrupts=(Interrupt(value='订单创建完成，等待哥们你的支付...', id='b63ac8c675718737076485b99fa18474'),))

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'parent_config': {'configurable': {'thread_id': '1',
                                    'checkpoint_ns': '',
                                    'checkpoint_id': '1f0c9f18-3c2d-640a-8000-4458b9da1ba0'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b')]},
 'metadata': {'source': 'loop',
              'step': 1,
              'parents': {}},
 'next': ['interrupt_node'],
 'tasks': [{'id': '2ff3e45a-a474-fb08-a15d-631aac7ea187',
            'name': 'interrupt_node',
            'interrupts': ({'value': '订单创建完成，等待哥们你的支付...',
                            'id': 'b63ac8c675718737076485b99fa18474'},),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'interrupt_node': {'chat': 1,
                    'messages': [{'role': 'human',
                                  'content': '你真有啊？'},
                                 {'role': 'ai',
                                  'content': '包有的，哥们！'}]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75')],
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 2,
              'parents': {}},
 'next': [],
 'tasks': []}

====================================================================================================
state:  StateSnapshot(values={'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'), AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'), HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'), AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75')], 'chat': 1}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2025-11-25T11:25:58.338770+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}}, tasks=(), interrupts=())

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}},
 'parent_config': {'configurable': {'thread_id': '1',
                                    'checkpoint_ns': '',
                                    'checkpoint_id': '1f0c9f18-3c2e-644a-8001-b7a54ac1e594'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75')],
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 2,
              'parents': {}},
 'next': ['interrupt_node'],
 'tasks': [{'id': '672d0452-4f2d-cf6b-5499-3fb9c79845b8',
            'name': 'interrupt_node',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'interrupt_node': {'interrupt': 1,
                    'messages': [{'role': 'ai',
                                  'content': '我付完款了，哥们，真不赖!'}]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c34-64da-8003-a7facb4cab51'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c31-685c-8002-6a11cc8bc171'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75'),
                         AIMessage(content='我付完款了，哥们，真不赖!', additional_kwargs={}, response_metadata={}, id='d3797fd1-c200-45dc-b19a-b92ef187f06e')],
            'interrupt': 1,
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 3,
              'parents': {}},
 'next': ['data_node'],
 'tasks': [{'id': '07eb0f07-5c1f-58f5-e985-33092ee626a9',
            'name': 'data_node',
            'interrupts': (),
            'state': None}]}

****************************************************************************************************
model:  updates
value: 
{'data_node': {'messages': [AIMessage(content='哥们已经在 🫵 给你发货了', additional_kwargs={}, response_metadata={}, id='run--50c0d268-00c9-4c7f-b076-663988e10803-0')]}}

****************************************************************************************************
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0c9f18-3c36-6212-8004-acae72ab008d'}},
 'parent_config': {'configurable': {'checkpoint_ns': '',
                                    'thread_id': '1',
                                    'checkpoint_id': '1f0c9f18-3c34-64da-8003-a7facb4cab51'}},
 'values': {'messages': [HumanMessage(content='哥们，我来买一张方大同的黑胶唱片', additional_kwargs={}, response_metadata={}, id='543ef59c-4258-415f-be19-2ace2f1fc3c5'),
                         AIMessage(content='下单成功，请支付...', additional_kwargs={}, response_metadata={}, id='4536ca70-bce2-4e75-8aa4-12882baa369b'),
                         HumanMessage(content='你真有啊？', additional_kwargs={}, response_metadata={}, id='ae1bf489-b931-47b0-8f36-aa6801677cbf'),
                         AIMessage(content='包有的，哥们！', additional_kwargs={}, response_metadata={}, id='d6a22cfc-754d-474d-9678-7bbd321d4a75'),
                         AIMessage(content='我付完款了，哥们，真不赖!', additional_kwargs={}, response_metadata={}, id='d3797fd1-c200-45dc-b19a-b92ef187f06e'),
                         AIMessage(content='哥们已经在 🫵 给你发货了', additional_kwargs={}, response_metadata={}, id='run--50c0d268-00c9-4c7f-b076-663988e10803-0')],
            'interrupt': 1,
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 4,
              'parents': {}},
 'next': [],
 'tasks': []}
```

## 问题

如果已经有了一个中断 Interrupt ，那么 command 跳转后执行就会出现问题

具体看代码吧

提了一个 [issue](https://github.com/langchain-ai/langgraph/issues/6534#issue-3693243458)

```python
from pprint import pprint

from langchain_core.language_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command


class State(MessagesState):
    interrupt: int
    chat: int


def payment_interrupt_node(state: State):
    message_list = []
    fake_message = FakeListChatModel(responses=["Order creation successful!"]).invoke("")
    message_list.append(fake_message)

    payload = interrupt("Payment Link...")

    data = payload.get("data")  # payment data
    message = payload.get("message")  # human message

    if data:
        message_list.append({"role": "ai", "content": data})
        return Command(
            goto="delivery",
            update={
                "interrupt": state.get("interrupt", 0) + 1,
                "messages": message_list
            }
        )

    if message:
        message_list.append({"role": "human", "content": message})
        if message == "normal conversation":
            message_list.append({"role": "ai", "content": "relpy message"})
            return {
                "chat": state.get("chat", 0) + 1,
                "messages": message_list
            }

        message_list.append({"role": "ai", "content": "some interrupt"})
        return Command(
            goto="another_interrupt_node",
            update={
                "chat": state.get("chat", 0) + 1,
                "messages": message_list
            }
        )

    # Error
    message_list.append({"role": "ai", "content": "Error!!!"})
    return {"messages": message_list}


def delivery(state: State):
    return {"messages": [{"role": "ai", "content": "Delivery message."}]}


def another_interrupt_node(state: State):
    # ...
    payload = interrupt("Send another interrupt(for some reason)")
    return {"messages": [
        {"role": "human", "content": str(payload)},
        {"role": "ai", "content": "never mind"},
    ]}


# build graph
graph = StateGraph(State)

graph.add_node(payment_interrupt_node)
graph.add_node(delivery)
graph.add_node(another_interrupt_node)

graph.add_edge(START, "payment_interrupt_node")
graph.add_edge("another_interrupt_node", END)
graph.add_edge("delivery", END)

checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

# create order and need payment
for model, value in graph.stream(
        {"messages": [{"role": "user", "content": "buy something"}]},
        stream_mode=["messages", "updates", "checkpoints"],
        config=config,
        debug=True,
):
    pass
    # print("model: ", model)
    # print("value: ")
    # pprint(value, sort_dicts=False, width=40)

print("+" * 100)

################################ Case: Pay Now ##########################################
# #
# # There is no problem with this case, so let's ignore it
# for model, value in graph.stream(
#         Command(resume={"data": "pay success"}),
#         stream_mode=["messages", "updates", "checkpoints"],
#         config=config,
#         debug=True,
# ):
#     pass
#     # print("model: ", model)
#     # print("value: ")
#     # pprint(value, sort_dicts=False, width=40)


################################ Case: Pay Later (without interrupt) ##########################################
# for model, value in graph.stream(
#         Command(resume={"message": "normal conversation"}),
#         stream_mode=["messages", "updates", "checkpoints"],
#         config=config,
#         debug=True,
# ):
#     pass
#     # print("model: ", model)
#     # print("value: ")
#     # pprint(value, sort_dicts=False, width=40)
#
# print("+" * 100)
#
# # then the user paid for the order
# for model, value in graph.stream(
#         Command(
#             goto="payment_interrupt_node",
#             resume={"data": "30$"}
#         ),
#         stream_mode=["messages", "updates", "checkpoints"],
#         config=config,
#         debug=True,
# ):
#     pass
#     # print("model: ", model)
#     # print("value: ")
#     # pprint(value, sort_dicts=False, width=40)


################################ Case: Pay Later (with interrupt) ##########################################
for model, value in graph.stream(
        Command(resume={"message": "ask some information"}),
        stream_mode=["messages", "updates", "checkpoints"],
        config=config,
        debug=True,
):
    pass
    # print("model: ", model)
    # print("value: ")
    # pprint(value, sort_dicts=False, width=40)

print("+" * 100)

# there have an interrupt
# at same time
# then the user paid for the order
for model, value in graph.stream(
        Command(
            goto="payment_interrupt_node",
            resume={"data": "30$"}
        ),
        stream_mode=["checkpoints"],
        config=config,
        debug=True,
):
    # pass
    print("model: ", model)
    print("value: ")
    pprint(value, sort_dicts=False, width=40)

print("+" * 100)

# here not go to the delivery node, i don't know why
state = graph.get_state(config=config)
print("state: ", state)

```

### 输出

```python
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f')]}
[updates] {'__interrupt__': (Interrupt(value='Payment Link...', id='c0fb60800beeed446c93ad2b8bf6664a'),)}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f')], '__interrupt__': (Interrupt(value='Payment Link...', id='c0fb60800beeed446c93ad2b8bf6664a'),)}
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f')]}
[updates] {'payment_interrupt_node': {'chat': 1, 'messages': [AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), {'role': 'human', 'content': 'ask some information'}, {'role': 'ai', 'content': 'some interrupt'}]}}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b')], 'chat': 1}
[updates] {'__interrupt__': (Interrupt(value='Send another interrupt(for some reason)', id='b51ab1eac2e8c57092a14fc1bce1e4b5'),)}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b')], 'chat': 1, '__interrupt__': (Interrupt(value='Send another interrupt(for some reason)', id='b51ab1eac2e8c57092a14fc1bce1e4b5'),)}

# CASE_1: another_interrupt_node
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b')], 'chat': 1}
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0d0de5-3edf-6edc-8001-cfb96672d6aa'}},
 'parent_config': {'configurable': {'thread_id': '1',
                                    'checkpoint_ns': '',
                                    'checkpoint_id': '1f0d0de5-3ed1-6760-8000-d80192f02970'}},
 'values': {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'),
                         AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'),
                         HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'),
                         AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b')],
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 1,
              'parents': {}},
 'next': ['another_interrupt_node',
          'payment_interrupt_node'],
 'tasks': [{'id': '082e1f9e-955b-fd99-852f-f6bb62d3c174',
            'name': 'another_interrupt_node',
            'interrupts': ({'value': 'Send '
                                     'another '
                                     'interrupt(for '
                                     'some '
                                     'reason)',
                            'id': 'b51ab1eac2e8c57092a14fc1bce1e4b5'},),
            'state': None},
           {'id': '36f1a7d2-f471-a97b-536c-a54d7e2862b2',
            'name': 'payment_interrupt_node',
            'interrupts': (),
            'state': None}]}
[updates] {'another_interrupt_node': {'messages': [{'role': 'human', 'content': "{'data': '30$'}"}, {'role': 'ai', 'content': 'never mind'}]}}
[updates] {'__interrupt__': (Interrupt(value='Payment Link...', id='bc79a0ab272d5ba9d239043028d345cd'),)}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b')], 'chat': 1, '__interrupt__': (Interrupt(value='Payment Link...', id='bc79a0ab272d5ba9d239043028d345cd'),)}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b'), HumanMessage(content="{'data': '30$'}", additional_kwargs={}, response_metadata={}, id='1a5a6f70-e3f0-4145-a3ac-5627616ecaa4'), AIMessage(content='never mind', additional_kwargs={}, response_metadata={}, id='2605030a-4040-4fbf-aa06-6497ef5123ae')], 'chat': 1}
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
state:  StateSnapshot(values={'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='098cc5e6-70d0-49ee-be9d-81e56bda1b9f'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--4167de44-b228-47b7-9b46-264abd1417a4'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='cf6ed29b-0595-4de6-ba9a-10bb537d7584'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d79434ae-2225-49d8-b371-598c7869f92b'), HumanMessage(content="{'data': '30$'}", additional_kwargs={}, response_metadata={}, id='180dc6ad-0733-4c92-92a2-cec63db41c42'), AIMessage(content='never mind', additional_kwargs={}, response_metadata={}, id='11da48f9-fe67-4b0f-ae4c-7a5db0bf4263')], 'chat': 1}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d0de5-3edf-6edc-8001-cfb96672d6aa'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-12-04T06:56:15.790857+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d0de5-3ed1-6760-8000-d80192f02970'}}, tasks=(PregelTask(id='082e1f9e-955b-fd99-852f-f6bb62d3c174', name='another_interrupt_node', path=('__pregel_pull', 'another_interrupt_node'), error=None, interrupts=(Interrupt(value='Send another interrupt(for some reason)', id='b51ab1eac2e8c57092a14fc1bce1e4b5'),), state=None, result={'messages': [{'role': 'human', 'content': "{'data': '30$'}"}, {'role': 'ai', 'content': 'never mind'}]}),), interrupts=(Interrupt(value='Send another interrupt(for some reason)', id='b51ab1eac2e8c57092a14fc1bce1e4b5'),))

# CASE_2: payment_interrupt_node
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='a50e744d-b5e8-4d95-9388-4e0598481a10'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--24a43d8e-20ea-4b1c-b6c4-0ef060565f53'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='506df1b9-42c4-4265-ac56-0b68979f3b41'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d8ff3231-dfb1-44af-b433-1ca237e4e7d7')], 'chat': 1}
model:  checkpoints
value: 
{'config': {'configurable': {'checkpoint_ns': '',
                             'thread_id': '1',
                             'checkpoint_id': '1f0d17d6-c2fc-6f1c-8001-ed3f54d18cdc'}},
 'parent_config': {'configurable': {'thread_id': '1',
                                    'checkpoint_ns': '',
                                    'checkpoint_id': '1f0d17d6-c2f4-6146-8000-26fe9ef64f93'}},
 'values': {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='a50e744d-b5e8-4d95-9388-4e0598481a10'),
                         AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--24a43d8e-20ea-4b1c-b6c4-0ef060565f53'),
                         HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='506df1b9-42c4-4265-ac56-0b68979f3b41'),
                         AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d8ff3231-dfb1-44af-b433-1ca237e4e7d7')],
            'chat': 1},
 'metadata': {'source': 'loop',
              'step': 1,
              'parents': {}},
 'next': ['another_interrupt_node',
          'payment_interrupt_node'],
 'tasks': [{'id': 'fcf8b6bc-0c4a-d439-88f2-f0bf6f160ced',
            'name': 'another_interrupt_node',
            'interrupts': ({'value': 'Send '
                                     'another '
                                     'interrupt(for '
                                     'some '
                                     'reason)',
                            'id': '1635cfd14c3a5b289c981707c2255da7'},),
            'state': None},
           {'id': '34ffbfb5-bd9c-7dae-c41b-6ffc01ac64c6',
            'name': 'payment_interrupt_node',
            'interrupts': (),
            'state': None}]}
[updates] {'payment_interrupt_node': {'interrupt': 1, 'messages': [AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--10a67ac2-ce12-447d-89c6-f04222b74c6d-0'), {'role': 'ai', 'content': '30$'}]}}
[updates] {'__interrupt__': (Interrupt(value='Send another interrupt(for some reason)', id='1635cfd14c3a5b289c981707c2255da7'),)}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='a50e744d-b5e8-4d95-9388-4e0598481a10'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--24a43d8e-20ea-4b1c-b6c4-0ef060565f53'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='506df1b9-42c4-4265-ac56-0b68979f3b41'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d8ff3231-dfb1-44af-b433-1ca237e4e7d7')], 'chat': 1, '__interrupt__': (Interrupt(value='Send another interrupt(for some reason)', id='1635cfd14c3a5b289c981707c2255da7'),)}
[values] {'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='a50e744d-b5e8-4d95-9388-4e0598481a10'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--24a43d8e-20ea-4b1c-b6c4-0ef060565f53'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='506df1b9-42c4-4265-ac56-0b68979f3b41'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d8ff3231-dfb1-44af-b433-1ca237e4e7d7'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--10a67ac2-ce12-447d-89c6-f04222b74c6d-0'), AIMessage(content='30$', additional_kwargs={}, response_metadata={}, id='83414148-d903-41f9-a726-9853cd82ff3e')], 'interrupt': 1, 'chat': 1}
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
state:  StateSnapshot(values={'messages': [HumanMessage(content='buy something', additional_kwargs={}, response_metadata={}, id='a50e744d-b5e8-4d95-9388-4e0598481a10'), AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--24a43d8e-20ea-4b1c-b6c4-0ef060565f53'), HumanMessage(content='ask some information', additional_kwargs={}, response_metadata={}, id='506df1b9-42c4-4265-ac56-0b68979f3b41'), AIMessage(content='some interrupt', additional_kwargs={}, response_metadata={}, id='d8ff3231-dfb1-44af-b433-1ca237e4e7d7')], 'chat': 1}, next=('another_interrupt_node',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d17d6-c2fc-6f1c-8001-ed3f54d18cdc'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-12-05T01:55:06.467500+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d17d6-c2f4-6146-8000-26fe9ef64f93'}}, tasks=(PregelTask(id='fcf8b6bc-0c4a-d439-88f2-f0bf6f160ced', name='another_interrupt_node', path=('__pregel_pull', 'another_interrupt_node'), error=None, interrupts=(Interrupt(value='Send another interrupt(for some reason)', id='1635cfd14c3a5b289c981707c2255da7'),), state=None, result=None),), interrupts=(Interrupt(value='Send another interrupt(for some reason)', id='1635cfd14c3a5b289c981707c2255da7'),))
```

### 分析

```python
# 关键在最后的这里会出现两个情况
Command(
  goto="payment_interrupt_node",
  resume={"data": "30$"}
)

CASE_1: another_interrupt_node

# 这里 another_interrupt_node 消耗了 resume 的 {"data": "30$"}
[updates] {'another_interrupt_node': {'messages': [{'role': 'human', 'content': "{'data': '30$'}"}, {'role': 'ai', 'content': 'never mind'}]}}
# 这里的 Interrupt 在 StateSnapshot 中不存在
[updates] {'__interrupt__': (Interrupt(value='Payment Link...', id='bc79a0ab272d5ba9d239043028d345cd'),)}



CASE_2: payment_interrupt_node

# 这里 payment_interrupt_node 消耗了 resume 的 {"data": "30$"}
[updates] {'payment_interrupt_node': {'interrupt': 1, 'messages': [AIMessage(content='Order creation successful!', additional_kwargs={}, response_metadata={}, id='lc_run--10a67ac2-ce12-447d-89c6-f04222b74c6d-0'), {'role': 'ai', 'content': '30$'}]}}
# 这里应该去 delivery node
[updates] {'__interrupt__': (Interrupt(value='Send another interrupt(for some reason)', id='1635cfd14c3a5b289c981707c2255da7'),)}

# 'interrupt': 1 也不存在与 StateSnapshot 中
```

