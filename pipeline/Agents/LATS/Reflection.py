import math
import os
from collections import deque
from typing import Optional
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import  BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
)

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv('.env')


from LLMs import get_llm
from Agents.LATS.OldfinTools import get_process_log_path

WRITE_ARTIFACTS = os.getenv("WRITE_ARTIFACTS", "false").lower() in {"1", "true", "yes"}

class Reflection(BaseModel):
    reflections: str = Field(
        description=
        f'''The critique and reflections on the sufficiency of the questions, superfluency and general quality of the response. The response must be accurate and backed by soures. Include full WEB-LINKS[THE FULL URL] next to the relevant information in each response IF POSSIBLE.'''
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task and the response is accurate and backed by sources including sources next to the relevant information in each response."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions is accurate and backed by data sources,including full WEB-LINKS[THE FULL URL] next to the relevant information in each response we can end the search.
        We can also end the result if the respoonse is NULL, or the response is along the lines of retrieval not possible, or incorrect ticker"""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            if WRITE_ARTIFACTS:
                log_path = get_process_log_path()
                with open(log_path, "a") as f:
                    for i in self.messages + [self.reflection.as_message()]:
                        if 'tool_calls' in i.additional_kwargs:
                            f.write("CALLING TOOLS NOW\n")
                            for j in i.additional_kwargs['tool_calls']:
                                f.write("Calling Tool ")
                                f.write(f"{j['function']['name']}\n")
                                f.write("Function has Arguments\n")
                                f.write(f"{j['function']['arguments']}\n\n")
                            f.write("Agent Tools RAW Output:\n")
                            f.write(f"{i.content}\n\n")
                        else:
                            f.write("Reflections Output\n")
                            f.write(f"{i.content}\n\n")

            
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reflect and grade the assistant response to the user question below, give better grade to the response that is more accurate ,informative, and contains more numerical data. Accuracy is the highest priority",
        ),
        (
            "system",
            "Do not call the same tool if the tool returns the message that it has failed or returned null, instead call another relevant tool to get the response.If you do not have any relevant tool to call, you have to use web search to get the detailed response.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)


@as_runnable
def reflection_chain(inputs) -> Reflection:
    llm = get_llm(
        model=inputs.get("_model"),
        provider=inputs.get("_provider"),
        role="lats",
    )
    reflection_llm_chain = (
        prompt
        | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
            run_name="Reflection"
        )
        | PydanticToolsParser(tools=[Reflection])
    )
    tool_choices = reflection_llm_chain.invoke(
        {
            "input": inputs.get("input"),
            "candidate": inputs.get("candidate"),
        }
    )
    reflection = tool_choices[0]
    if reflection.score >= 6:
        reflection.found_solution = True
    return reflection
