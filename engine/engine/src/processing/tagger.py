from engine.logger import make_logger
from engine.src.processing.base import StatementProcessor
from engine.config import Config
import asyncio
from typing import List
from tqdm.asyncio import tqdm as tqdm_async

logger = make_logger(__name__)

class StatementTagger(StatementProcessor):
    """Tag relevant parts of text using <start> and <end> tags"""

    def __init__(self, config: Config, tagging_prompt: str):
        super().__init__(config, tagging_prompt)

    async def _process_statement_async(self, statement: str) -> str:
        """Process a statement without chunking"""
        if not isinstance(statement, str) or not statement.strip():
            return ""
        system_prompt = "You are an international expert in macroeconomics and central bank policy."
        user_query = self.processing_prompt + "\n Part of the transcript: " + statement
        response = await self.llm_client.generate_text(system_prompt, user_query)
        return response
    
    def process_statements(self, statements: List[str]) -> List[str]:
        """Tag relevant parts in multiple statements"""
        self.logger.info(f"Tagging relevant parts in {len(statements)} texts")
        
        async def run_all():
            tasks = [self._process_statement_async(statement) for statement in statements]
            return await tqdm_async.gather(*tasks, desc="Tagging Statements")

        return asyncio.run(run_all())


if __name__ == "__main__":
    config = Config()

    df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/pc_data/summarized_data.csv")
    text = df['text'].iloc[-1]

    tagger = StatementTagger(config, "")
    results = tagger.process_statements([text])

    results = {'result': results}
    with open("results.json", "w") as f:
        json.dump(results, f)

    