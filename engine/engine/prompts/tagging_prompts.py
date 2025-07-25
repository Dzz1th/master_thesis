import pandas as pd 
import asyncio
from tqdm.asyncio import tqdm   

employment_tagging_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    Your task is to analyze a part of FED press conference transcript and find statements that help us to understand the current state of the labour market orits potential future state and its dynamics.
    By statement we imply a solid and coherent part of the text. It should not be a a random subset of a sentence.
    Statemte should convey useful information that helps us to understand either the current state of the labour market or its potential future state, its dynamics and its perspectives.

    **Guidelines:**
    - Look for sentences that contain any of the following keywords or phrases:  
      **"labor market," "employment," "jobs," "unemployment," "job gains," "wage growth," "labor force participation," "maximum employment," "tight labor markets," "worker shortages," or "underemployment." and similar! This is not an exhaustive list!**
    - Also tag statements where FED comments on the current conditions and the state of the labour market and employment.
    - Place <start> tag before the relevant statement and <end> tag after it. Do not place tags in the middle of a word.
    - If you have a statement that discuss labour market and something else - still tag this statement.
    - Return the complete original text with only the relevant parts tagged.
    - If no relevant employment statements are found in the chunk, return the original text without any tags.
    - If you find an important statement from the FED chairman in the answer to some question and the question provides important context about the state of labour market and employment - tag the question as well.
    
    Examples:
    - "The labor market is tight, and we are seeing wage growth." - tag this statement
    - "Our monetary policy actions are guided by our dual mandate to promo<start>te maximum\nemployment<end> and stable prices for the American people." - wrong tagging, because <start> and <end> are in the middle of a word and because it is doesn't convey any useful information.

    Do not miss any statements that reflect FED's view on the labor market and employment, we need all the statements that are important to determine the FED's view and guidance on the labor market and employment.

    **Keep all of the original text intact, just add <start> and <end> tags around relevant employment-related parts.**
"""

inflation_tagging_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    Your task is to analyze a part of FED press conference transcript and find statements that help us to understand the current state of inflation or its potential future state and its dynamics.
    By statement we imply a solid and coherent part of the text. It should not be a random subset of a sentence.
    Statement should convey useful information that helps us to understand either the current state of inflation or its potential future state, its dynamics and its perspectives.

    **Guidelines:**
    - Look for sentences that contain any of the following keywords or phrases:  
      **"inflation," "price stability," "price pressures," "inflation expectations," "cost pressures," "inflation target," "2%," "core inflation," "headline inflation." and similar! This is not an exhaustive list!**
    - Also tag statements where FED comments on the current conditions and the state of inflation and price stability.
    - Place <start> tag before the relevant statement and <end> tag after it. Do not place tags in the middle of a word.
    - If you have a statement that discuss inflation and something else - still tag this statement.
    - Return the complete original text with only the relevant parts tagged.
    - If no relevant inflation statements are found in the chunk, return the original text without any tags.
    - If you find an important statement from the FED chairman in the answer to some question and the question provides important context about the state of inflation and price stability - tag the question as well.
    
    Examples:
    - "Inflation remains above our 2 percent target, though it has eased over the past year." - tag this statement
    - "Our monetary policy actions are guided by our dual mandate to promote maximum employment and sta<start>ble prices<end> for the American people." - wrong tagging, because <start> and <end> are in the middle of a word and because it doesn't convey any useful information.

    Do not miss any statements that reflect FED's view on inflation and price stability, we need all the statements that are important to determine the FED's view and guidance on inflation and price stability.

    **Keep all of the original text intact, just add <start> and <end> tags around relevant inflation-related parts.**
"""

forward_guidance_tagging_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand whether the FED is providing clear and explicit forward guidance or unclear and implicit forward guidance.

    Your task is to analyze a part of a transcript and tag statements that provide forward guidance using <start> and <end> tags.

    Return the complete original text with only the relevant parts tagged. Place <start> tag before each relevant part and <end> tag after it.

    Guidelines:
    1. Interest Rate Path
        - Explicit Forward Guidance:
            - Look for any sentences mentioning a specific timeline or conditions under which the Fed will raise or lower interest rates. For example, "We do not expect to raise rates until at least next year," "We will begin lowering rates once inflation has fallen to 2% for at least three months."
        -Implicit Forward Guidance:
            - Identify statements about rates that are open-ended or conditional, but lack details about timing or thresholds. For example "We will adjust policy as data evolve," "We remain data-dependent and will decide on a meeting-by-meeting basis."
    2. Balance Sheet Policy
        - Explicit Forward Guidance:
            - Watch for direct mentions of the pace or duration of quantitative tightening (QT) or expansions (QE). For example, "We will continue to reduce our holdings at $60 billion per month until mid-2025."
        -Implicit Forward Guidance:
            - Note any references to the balance sheet that do not include a specific plan or timeframe. "We will monitor market conditions and adjust our balance sheet approach if necessary."
    3. Conditions / Thresholds for Policy Changes
        - Clear Conditions:
            - Identify any statements describing exact economic thresholds (e.g., unemployment rate, inflation rate) for policy shifts. For example, "We will maintain rates until inflation is on track to moderately exceed 2% for some time."
        - Generic Conditions:
            - Flag statements that mention broad considerations but do not specify numeric targets or time frames. For example: "Our decisions will be guided by overall economic conditions and financial stability concerns."
    4. Tone in Q&A or Additional Remarks
        - Strength of Language:
            - Highlight strong phrases indicating commitment to a future path ("We are determined to keep rates at current levels for the next few quarters") versus hedged language ("We may consider pausing if growth slows, but it's too soon to say").
            - Collect any qualifiers that weaken clarity: "if needed," "depending on the data," "we remain flexible," "we haven't decided yet."
            - Capture any unambiguous promises or timelines: "We will keep rates near zero until the labor market has fully recovered."

    **Keep all of the original text intact, just add <start> and <end> tags around relevant forward guidance parts. If no relevant statements are found, return the original text without any tags.**
"""

interest_rate_tagging_prompt = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze a part of FED press conference transcript and find statements that reflects FED's guidance and view on the future path of interest rates.
    By statement we imply a solid and coherent part of the text. It should not be a random subset of a sentence.
    Statement should convey useful information that helps us to understand the FED's view on the future path of interest rates, its current and/or future state and dynamics.

    **Guidelines:**
    - Look for sentences that contain any of the following keywords or phrases:  
      **"rate cuts," "rate hikes," "interest rates," "monetary policy," "tightening," "raising," "increasing," "lowering," "easing," "pausing," "policy rate," "federal funds rate." and similar! This is not an exhaustive list!**
    - Also tag statements where FED comments on the direction of rate moves and conditions for future rate changes.
    - Place <start> tag before the relevant statement and <end> tag after it. Do not place tags in the middle of a word.
    - If you have a statement that discuss interest rates and something else - still tag this statement.
    - Return the complete original text with only the relevant parts tagged.
    - If no relevant interest rate statements are found in the chunk, return the original text without any tags.
    - If you find an important statement from the FED chairman in the answer to some question and the question provides important context about interest rates - tag the question as well.
    
    Examples:
    - "We do not expect to raise rates until at least next year." - tag this statement
    - "We will begin lowering rates once inflation has fallen to 2% for at least three months." - tag this statement
    - "Our monetary policy is cur<start>rently restrictive<end> and we will adjust as needed." - wrong tagging, because <start> and <end> are in the middle of a word.

    Do not miss any statements that reflect FED's view on the future path of interest rates, we need all the statements that are important to determine the FED's view and guidance on interest rates.

    **Keep all of the original text intact, just add <start> and <end> tags around relevant interest rate parts.**
"""

balance_sheet_tagging_prompt = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze a part of FED press conference transcript and find statements that reflects FED's guidance and view on the future path of its balance sheet.
    By statement we imply a solid and coherent part of the text. It should not be a random subset of a sentence.
    Statement should convey useful information that helps us to understand the FED's view on balance sheet policies, including asset purchases (QE) and quantitative tightening (QT).

    **Guidelines:**
    - Look for sentences that contain any of the following keywords or phrases:  
      **"balance sheet," "asset purchases," "quantitative easing," "QE," "quantitative tightening," "QT," "holdings," "securities," "treasuries," "MBS," "mortgage-backed securities," "runoff." and similar! This is not an exhaustive list!**
    - Also tag statements where FED comments on the current conditions and future plans related to their balance sheet policy.
    - Place <start> tag before the relevant statement and <end> tag after it. Do not place tags in the middle of a word.
    - If you have a statement that discuss balance sheet policy and something else - still tag this statement.
    - Return the complete original text with only the relevant parts tagged.
    - If no relevant balance sheet statements are found in the chunk, return the original text without any tags.
    - If you find an important statement from the FED chairman in the answer to some question and the question provides important context about balance sheet policy - tag the question as well.
    
    Examples:
    - "We will continue to reduce our holdings at $60 billion per month until mid-2025." - tag this statement
    - "We will monitor market conditions and adjust our balance sheet approach if necessary." - tag this statement
    - "The Committee's approach to reducing the size of the Fe<start>deral Reserve's balance sheet<end>." - wrong tagging, because <start> and <end> are in the middle of a word.

    Do not miss any statements that reflect FED's view on balance sheet policy, we need all the statements that are important to determine the FED's view and guidance on balance sheet trajectory.

    **Keep all of the original text intact, just add <start> and <end> tags around relevant balance sheet parts.**
"""

unified_macro_tagging_prompt = """
Your task is to analyze a part of a FED press conference transcript and find statements relevant to FOUR key macroeconomic topics:
1.  **Employment**: Labor market conditions, job growth, unemployment, wage dynamics, etc.
2.  **Inflation**: Price stability, inflation rates, price pressures, inflation expectations, etc.
3.  **Interest Rates**: The future path of interest rates, monetary policy stance (tightening/easing), federal funds rate, etc.
4.  **Balance Sheet**: The FED's asset holdings, quantitative easing (QE), quantitative tightening (QT), securities, etc.

For each relevant statement, you must enclose it with topic-specific tags as detailed below. The original text must be preserved in its entirety.

**General Guidelines (Applicable to ALL topics):**
-   A "statement" is a solid and coherent part of the text that conveys useful information about the topic. It should NOT be a random subset of a sentence or a phrase that doesn't offer substantial insight.
-   Return the complete original text, with relevant parts tagged as specified for each topic.
-   If no statements relevant to ANY of the topics are found, return the original text without any tags.
-   If an important statement from the FED chairman is in an answer to a question, and the question itself provides crucial context for one of the topics, tag the relevant part of the question as well, using the appropriate topic tags.
-   Ensure tags are NOT placed in the middle of a word. Tags must surround whole words or phrases.
-   A single piece of text might be relevant to multiple topics. Tag each relevant segment for its specific topic. It is possible for tags to be adjacent. For example, a sentence discussing inflation and then its impact on interest rates could have an inflation tag followed by an interest rate tag.
-   Strive to make tags as granular as possible. If different parts of a longer sentence relate to different topics, use separate tags for each part.
-   Avoid tagging overly generic phrases or mentions that don't provide specific insight into the FED's views or assessments for that topic (e.g., a generic mention of "stable prices" without further context on inflation may not be useful).

---

**Topic 1: Employment**
*   **Focus**: Understand the current state of the labor market, its potential future state, and its dynamics.
*   **Keywords/Phrases (Non-Exhaustive)**: "labor market," "employment," "jobs," "unemployment," "job gains," "wage growth," "labor force participation," "maximum employment," "tight labor markets," "worker shortages," "underemployment." Also include specific comments on current labor market conditions or outlook.
*   **Tag Format**: `<employment_start>relevant employment statement<employment_end>`
*   **Example**: "The <employment_start>labor market remains strong with continued job gains<employment_end>."

---

**Topic 2: Inflation**
*   **Focus**: Understand the current state of inflation, its potential future state, and its dynamics.
*   **Keywords/Phrases (Non-Exhaustive)**: "inflation," "price stability," "price pressures," "inflation expectations," "cost pressures," "inflation target," "2%," "core inflation," "headline inflation." Also include specific comments on current inflation conditions, risks, or progress towards price stability goals.
*   **Tag Format**: `<inflation_start>relevant inflation statement<inflation_end>`
*   **Example**: "<inflation_start>Inflation has shown signs of easing but remains above our target<inflation_end>."

---

**Topic 3: Interest Rates**
*   **Focus**: Understand the FED's guidance and view on the future path of interest rates and overall monetary policy stance.
*   **Keywords/Phrases (Non-Exhaustive)**: "rate cuts," "rate hikes," "interest rates," "monetary policy," "tightening," "raising," "increasing," "lowering," "easing," "pausing," "policy rate," "federal funds rate," "restrictive stance." Also include comments on the direction of rate moves and conditions for future rate changes.
*   **Tag Format**: `<interest_rate_start>relevant interest rate statement<interest_rate_end>`
*   **Example**: <interest_rate_start> "We anticipate that further rate increases may be appropriate to temper inflation<interest_rate_end>."

---

**Topic 4: Balance Sheet**
*   **Focus**: Understand the FED's guidance and view on the future path of its balance sheet, including asset purchases (QE) and quantitative tightening (QT).
*   **Keywords/Phrases (Non-Exhaustive)**: "balance sheet," "asset purchases," "quantitative easing," "QE," "quantitative tightening," "QT," "holdings," "securities," "treasuries," "MBS," "mortgage-backed securities," "runoff." Also include specific comments on current conditions and future plans for balance sheet policy.
*   **Tag Format**: `<balance_sheet_start>relevant balance sheet statement<balance_sheet_end>`
*   **Example**: "<balance_sheet_start> The Committee plans to continue reducing its holdings of Treasury securities and agency mortgage-backed securities<balance_sheet_end>."

---

**Combined Example:**
Consider the text: "The labor market is very strong, and inflation remains a concern, so we are considering further increases in the federal funds rate and will continue our balance sheet runoff."

A possible tagged output:
"<employment_start> The labor market is very strong<employment_end>, and <inflation_start>inflation remains a concern<inflation_end>, so <interest_rate_start> we are considering further increases in the federal funds rate<interest_rate_end> and  <balance_sheet_start> will continue our balance sheet runoff<balance_sheet_end>."

**Important Final Instructions:**
-   Do not miss any significant statements reflecting the FED's view on these four topics.
-   Keep ALL of the original text intact. Only add the specified topic-specific tags around relevant parts.
-   If a statement is relevant to multiple topics, ensure it is appropriately tagged for each, even if that means tags are adjacent or one encompasses another (though tags of the exact same type should not be nested).
"""

if __name__ == "__main__":
    data = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv")
    texts = data['sentiment_summary'].tolist()

    # Example of how to use these tagging prompts
    # Results would contain the original text with relevant parts tagged with <start> and <end>
    # results = asyncio.run(tqdm.gather(*[summarize_transcript(employment_tagging_prompt, text) for text in texts]))
    # data['employment_tagged'] = results
    # data.to_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/tagged_data.csv", index=False) 