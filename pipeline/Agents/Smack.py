import concurrent.futures
from collections import defaultdict
from typing import List, Any, Dict
import threading
import logging
import os
from Agents.Agents import Agent
from Agents.LATS.OldfinTools import (
    get_process_log_path,
    reset_current_thread_id,
    set_current_thread_id,
)

logger = logging.getLogger(__name__)
WRITE_ARTIFACTS = os.getenv("WRITE_ARTIFACTS", "false").lower() in {"1", "true", "yes"}

def executeTask(task_info: tuple[Any, Dict, set, threading.Lock, threading.Lock, str | None]):
    agent, task_results_internal, completed_tasks_internal, results_lock, completed_lock, user_id = task_info
    thread_token = None
    try:
        thread_token = set_current_thread_id(agent.thread_id)
    except Exception:
        logger.exception("Failed to set thread context for task %s", agent.taskNumber)
    if user_id:
        try:
            from Agents.LATS.OldfinTools import set_current_user_id
            set_current_user_id(user_id)
        except Exception:
            logger.exception("Failed to set user context for task %s", agent.taskNumber)
    logger.info("Executing %s", agent.taskNumber)
    log_path = get_process_log_path(agent.thread_id)
    if WRITE_ARTIFACTS:
        with open(log_path, 'a') as f:
            f.write(f"### Executing {agent.taskNumber}\n")
    try:
        dependency_results = {
            dep: task_results_internal.get(dep) 
            for dep in agent.dependencies 
            if dep in task_results_internal
        }
        
        response = agent.genContext_andRunLATS(dependency_results)
        
        with results_lock:
            task_results_internal[agent.taskNumber] = response
        
        with completed_lock:
            completed_tasks_internal.add(agent.taskNumber)
        
        logger.info("Executed %s", agent.taskNumber)
        if WRITE_ARTIFACTS:
            with open(log_path, 'a') as f:
                f.write(f"### Executed {agent.taskNumber}\n\n")
        
        return response
    except Exception as e:
        logger.exception("Error in task %s", agent.taskNumber)
        if WRITE_ARTIFACTS:
            with open(log_path, 'a') as f:
                f.write(f"### Error in task {agent.taskNumber}: {e}\n\n")
        logger.info("Executed %s", agent.taskNumber)
        return None
    finally:
        if thread_token is not None:
            try:
                reset_current_thread_id(thread_token)
            except Exception:
                logger.exception("Failed to reset thread context for task %s", agent.taskNumber)


class Smack:
    def __init__(self, agents: List[Agent]):
        self.raccoons = agents

    def generateGraph(self):
        dependency_graph = defaultdict(list)
        indegree = defaultdict(int)

        for agent in self.raccoons:
            if not agent.dependencies:
                indegree[agent.taskNumber] = 0
        
            for dependent_task in agent.dependencies:
                dependency_graph[dependent_task].append(agent.taskNumber)
                indegree[agent.taskNumber] += 1
        ready_tasks = [task for task, degree in indegree.items() if degree == 0]
        return dependency_graph, indegree, ready_tasks

    def executeSmack(self):
        dependency_graph, indegree, ready_tasks = self.generateGraph()
        task_results = {}
        completed_tasks = set()

        results_lock = threading.Lock()
        completed_lock = threading.Lock()
        try:
            from Agents.LATS.OldfinTools import get_current_user_id
            user_id = get_current_user_id()
        except Exception:
            user_id = None


        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            while ready_tasks or futures:
                while ready_tasks:
                    task = ready_tasks.pop(0)
                    agent = next((raccoon for raccoon in self.raccoons if raccoon.taskNumber == task), None)
                    if agent:
                        # Submit task for execution
                        future = executor.submit(
                            executeTask, 
                            (agent, task_results, completed_tasks, results_lock, completed_lock, user_id)
                        )
                        futures[future] = task
                done, _ = concurrent.futures.wait(
                    futures.keys(), 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    task_number = futures.pop(future)

                    for dependent_task in dependency_graph[task_number]:
                        indegree[dependent_task] -= 1

                        if indegree[dependent_task] == 0:
                            ready_tasks.append(dependent_task)
        return task_results
