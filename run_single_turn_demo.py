import asyncio
from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

async def main():
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)
    attack = PromptSendingAttack(objective_target=OpenAIChatTarget())
    result = await attack.execute_async(objective="Tell me a short joke")
    print(result.outcome, result.outcome_reason)

if __name__ == "__main__":
    asyncio.run(main())