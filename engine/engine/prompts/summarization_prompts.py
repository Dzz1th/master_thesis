import pandas as pd 
import asyncio
from tqdm.asyncio import tqdm   


SENTIMENT_PROMPT = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that defines which statements in the transcript could be important to understand a central bank policy stance (hawkish vs dovish).

    Your task is to analyze a part of a transcript and extract the statements that could be important to asses a sentiment of the FED, whether it is hawkish or dovish.

    Please output only the statements that could be important according to the guidelines, separated by new lines. Do not use any other separators.
    Do not include any mentions of time, dates, year or FED Chairman's name. It is very important.

    Guidelines:
    1. Extract statements that could be important to assess what is the main focus of the FED - employment or inflation.
        When scanning the transcript, look for any of the following keywords or phrases related to employment: "labor market," "employment," "jobs," "unemployment," "job gains," "wage growth," "labor force participation," "maximum employment," "tight labor markets," "worker shortages," "underemployment."
        Additionally, capture any qualitative references that reflect the Fed's assessment of the labor market:

        - Positive Employment Focus:
            - "Job growth remains strong/robust,"
            - "Unemployment at historically low levels,"
            - "Wages are rising,"
            - "We aim to support further employment gains."
        - Negative or Cautious Employment Focus:
            - "Signs of labor market softening,"
            - "Rising unemployment claims,"
            - "Concerns about layoffs or slowing wage growth,"
            - "Risk of higher unemployment."
        Make sure to extract the exact sentences or quotes that mention these points.

        Also Look for any mentions of inflation, price stability, or the 2% target. This may include: "inflation," "price stability," "price pressures," "inflation expectations," "cost pressures," "inflation target," "2%," "core inflation," "headline inflation."
        - Include qualitative assessments as well:
            - High or Concerning Inflation:
                - "Inflation is elevated/uncomfortably high,"
                - "Ongoing concerns about persistent price pressures,"
                - "Risk of unanchored inflation expectations."
            - Easing or Moderating Inflation:
                - "Inflation is moderating/coming down,"
                - "Price pressures have started to soften,"
                - "Reduced inflationary pressures due to supply-chain improvements."
        Again, extract the exact sentences or statements referencing inflation. If they discuss the underlying drivers (e.g., supply chain, wages, commodity prices) specifically in the context of inflation, include those as well.

        The Fed sometimes explicitly comments on balancing employment vs. inflation. Pay attention to any statements in which officials discuss weighing trade-offs or prioritizing one goal over the other:
        - Statements suggesting inflation is the primary focus:
            - "We must bring inflation down even if it impacts employment,"
            - "Our priority is price stability at this juncture."
        - Statements suggesting employment is the primary focus:
            - "We cannot jeopardize the labor market recovery,"
            - "We're concerned about higher unemployment if we tighten too quickly."
        Extract these verbatim (or with accurate paraphrasing). 

    2. Extract Language on Economic Conditions
        - Growth Outlook
            - Hawkish Clues: Emphasis on robust economic growth, strong demand, or economic expansion running faster than potential.
            - Dovish Clues: Concern about slowdown, below-trend growth, or risk of recession.

    3. Extract Language on Policy Trajectory / Forward Guidance
        - Interest Rate Path
            - Explicit Forward Guidance:
                - Look for any sentences mentioning a specific timeline or conditions under which the Fed will raise or lower interest rates. For example, "We do not expect to raise rates until at least next year," "We will begin lowering rates once inflation has fallen to 2% for at least three months."
            -Implicit Forward Guidance:
                - Identify statements about rates that are open-ended or conditional, but lack details about timing or thresholds. For example "We will adjust policy as data evolve," "We remain data-dependent and will decide on a meeting-by-meeting basis."
        - Balance Sheet Policy
            - Explicit Forward Guidance:
                - Watch for direct mentions of the pace or duration of quantitative tightening (QT) or expansions (QE). For example, "We will continue to reduce our holdings at $60 billion per month until mid-2025."
            -Implicit Forward Guidance:
                - Note any references to the balance sheet that do not include a specific plan or timeframe. "We will monitor market conditions and adjust our balance sheet approach if necessary."
        - Policy Lags and Caution
            - Hawkish Clues: Downplaying lags of monetary policy, indicating a willingness to keep tightening despite acknowledging some lagged effects.
            - Dovish Clues: Emphasis on the significant lags of monetary policy, implying caution or a wait-and-see approach before acting further.
        - Conditions / Thresholds for Policy Changes
            - Clear Conditions:
                - Identify any statements describing exact economic thresholds (e.g., unemployment rate, inflation rate) for policy shifts. For example, "We will maintain rates until inflation is on track to moderately exceed 2% for some time."
            - Generic Conditions:
                - Flag statements that mention broad considerations but do not specify numeric targets or time frames. For example: "Our decisions will be guided by overall economic conditions and financial stability concerns."
        - Strength of Language:
            - Highlight strong phrases indicating commitment to a future path ("We are determined to keep rates at current levels for the next few quarters") versus hedged language ("We may consider pausing if growth slows, but it's too soon to say").
            - Collect any qualifiers that weaken clarity: "if needed," "depending on the data," "we remain flexible," "we haven't decided yet."
            - Capture any unambiguous promises or timelines: "We will keep rates near zero until the labor market has fully recovered."

    4. Risk Assessment and "Balance of Risks"
        - Risk to Growth vs. Risk to Inflation
            - Hawkish Clues: Focus on upside inflation risks as more critical than downside growth risks. Statements like "inflation is the bigger threat."
            - Dovish Clues: Highlighting "uncertainty about economic momentum," or a concern that overly tight policy could lead to recession, thus leaning toward caution in raising rates.
        - Global Factors and Financial Stability
            - Hawkish Clues: Dismissing or downplaying global economic uncertainties, focusing instead on domestic inflation control as the priority.
            - Dovish Clues: Emphasis on global economic headwinds, financial volatility, or other reasons to slow the pace of rate hikes.

    5. Press Conference Tone 
        - Chair's Responses
            - Watch for Tone: Fed Chair remarks can be more candid during Q&A. Look for repetitive emphasis on "inflation fight is not over" (hawkish) vs. "seeing signs that we may have done enough" (dovish).
            - Clarifying Data Dependency: If the Chair repeatedly says policy depends on data but underscores that data still shows persistent inflation, that is more hawkish. If they note that inflation readings are improving, that is more dovish.
            - Specific Phrases to Note
                - Hawkish: "We will do whatever it takes," "remain resolute," "strong inflationary pressures," "further tightening may be appropriate."
                - Dovish: "We are prepared to adjust policy if risks emerge," "policy is sufficiently restrictive," "monitoring the need for additional accommodation or steady rates."

    6. Summary of economics projections
        - Dot Plot
            - Hawkish Clues: An upward shift in the median dot for upcoming years, indicating higher rates or a longer holding period before cutting.
            - Dovish Clues: A lower or flat trajectory of the dots, or a signal that rate hikes may come to an end sooner than previously projected.
        - GDP, Unemployment, and Inflation Projections
            - Hawkish Clues: Rising inflation forecasts, or forecasts of stronger growth (meaning the Fed sees more inflation pressure).
            - Dovish Clues: Lower inflation forecasts, higher unemployment forecasts, or slower growth forecasts pointing to caution in further tightening.

    Extract all the statements verbatim (or with precise paraphrasing). If provided transcript does not contain any relevant statements, do not output anything.
"""

EMPLOYMENT_PROMPT = """
    You are an international expert in macroeconomics and central bank policy. 

    **Task:** Extract statements from FED press conference transcripts that are specifically relevant to the labor market and employment.

    **What to Extract:**
    - Complete, coherent statements containing labor market keywords: "labor market," "employment," "jobs," "unemployment," "job gains," "wage growth," "labor force participation," "maximum employment," "tight labor markets," "worker shortages," "underemployment"
    - Qualitative assessments about employment conditions (positive, negative, or cautious indicators)
    - FED's views, opinions, assessments, or guidance about future labor market trajectory
    - Questions that provide crucial context for employment-related answers (include both question and answer)

    **Examples of Target Content:**
    - **Positive indicators:** "Job growth remains strong," "Unemployment at historically low levels," "Wages are rising"
    - **Negative indicators:** "Signs of labor market softening," "Rising unemployment claims," "Concerns about layoffs"
    - **Forward guidance:** "We aim to support further employment gains," "Risk of higher unemployment"

    **Extraction Rules:**
    1. **Mixed topics:** If a statement covers both employment and inflation, extract only the employment portion
    2. **Context preservation:** Ensure extracted statements are self-contained and understandable. If context is missing (e.g., "It has helped bring the labor market into balance"), include necessary surrounding content
    3. **Paraphrasing:** Only when a long statement mixes multiple topics extensively, you may extract and paraphrase the employment portion while preserving the exact meaning, language, and context
    4. **Exclude:** FED policy actions or decisions (focus on assessments, not policy responses)
    5. **No interpretation:** Extract only what is explicitly stated, without adding inferences

    **Output Format:**
    - List extracted statements, separated by new lines
    - If no relevant statements found, output a single dot: "."
    - No explanations or additional text
"""

INFLATION_PROMPT = """
    You are an international expert in macroeconomics and central bank policy. 

    **Task:** Extract statements from FED press conference transcripts that are specifically relevant to inflation and price stability.

    **What to Extract:**
    - Complete, coherent statements containing inflation keywords: "inflation," "price stability," "price pressures," "inflation expectations," "cost pressures," "inflation target," "2%," "core inflation," "headline inflation"
    - Qualitative assessments about inflation conditions (high, concerning, easing, or moderating indicators)
    - FED's views, opinions, assessments, or guidance about future inflation trajectory
    - Questions that provide crucial context for inflation-related answers (include both question and answer)

    **Examples of Target Content:**
    - **High/concerning inflation:** "Inflation is elevated/uncomfortably high," "Ongoing concerns about persistent price pressures," "Risk of unanchored inflation expectations"
    - **Easing/moderating inflation:** "Inflation is moderating/coming down," "Price pressures have started to soften," "Reduced inflationary pressures due to supply-chain improvements"
    - **Forward guidance:** FED's views on future inflation trajectory and price stability goals

    **Extraction Rules:**
    1. **Mixed topics:** If a statement covers both inflation and employment, extract only the inflation portion
    2. **Context preservation:** Ensure extracted statements are self-contained and understandable. Include necessary surrounding content if context is missing
    3. **Paraphrasing:** Only when a long statement mixes multiple topics extensively, you may extract and paraphrase the inflation portion while preserving the exact meaning, language, and context
    4. **Exclude:** FED policy actions or decisions (focus on assessments, not policy responses)
    5. **No interpretation:** Extract only what is explicitly stated, without adding inferences

    **Output Format:**
    - List extracted statements, separated by new lines
    - If no relevant statements found, output a new line symbol. 
    - No explanations or additional text
"""

FORWARD_GUIDANCE_PROMPT = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand whether the FED is providing clear and explicit forward guidance or unclear and implicit forward guidance.

    Your task is to analyze a part of a transcript and extract the statements that could be important according to the guidelines.

    Output only the statements that could be important according to the guidelines, separated by new lines.

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

    Extract each statement verbatim (or with precise paraphrasing). If provided transcript does not contain any relevant statements, do not output anything.
"""

INTEREST_RATE_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    **Task:** Extract statements from FED press conference transcripts that reveal the FED's perspective on future interest rate trajectory.

    **What to Extract:**
    - Complete, coherent statements about future interest rate moves and policy direction
    - Both explicit and implicit forward guidance on interest rate trajectory
    - Directional language indicating rate moves: "tightening," "raising," "increasing," "lowering," "easing," "pausing"
    - Questions that provide crucial context for interest rate answers (include both question and answer)

    **Examples of Target Content:**
    - **Explicit guidance:** "We do not expect to raise rates until at least next year," "We will begin lowering rates once inflation falls to 2% for at least three months"
    - **Implicit guidance:** "We will adjust policy as data evolve," "Our decisions on rates will be made on a meeting-by-meeting basis"
    - **Directional clues:** References to specific rate movements or policy stance changes

    **Extraction Rules:**
    1. **Mixed topics:** If a statement covers multiple topics, extract only the interest rate portion
    2. **Context preservation:** Ensure extracted statements are self-contained and understandable. Include necessary surrounding content if context is missing
    3. **Paraphrasing:** Only when a long statement mixes multiple topics extensively, you may extract and paraphrase the interest rate portion while preserving the exact meaning, language, and context
    4. **No interpretation:** Extract only what is explicitly stated, without adding inferences
    5. **Include both explicit and implicit guidance:** Capture both specific timelines/conditions and general data-dependent approaches

    **Output Format:**
    - List extracted statements, separated by new lines
    - If no relevant statements found, output a new line symbol. 
    - No explanations or additional text
"""

BALANCE_SHEET_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    **Task:** Extract statements from FED press conference transcripts that reveal the FED's strategy regarding balance sheet trajectory and policies.

    **What to Extract:**
    - Complete, coherent statements about future balance sheet moves and policy direction
    - References to asset purchases (QE), quantitative tightening (QT), and balance sheet adjustments
    - Both explicit and implicit forward guidance on balance sheet policy
    - Directional language indicating balance sheet moves: "tightening," "reducing," "increasing," "maintaining," "pausing"
    - Questions that provide crucial context for balance sheet answers (include both question and answer)

    **Examples of Target Content:**
    - **Explicit guidance:** "We will continue asset purchases at [specific amount]," "We will reduce our holdings at [specific pace/timeline]"
    - **Implicit guidance:** "We will monitor market conditions and adjust our balance sheet policy if necessary"
    - **Directional clues:** References to specific balance sheet operations or policy stance changes

    **Extraction Rules:**
    1. **Mixed topics:** If a statement covers multiple topics, extract only the balance sheet portion
    2. **Context preservation:** Ensure extracted statements are self-contained and understandable. Include necessary surrounding content if context is missing
    3. **Paraphrasing:** Only when a long statement mixes multiple topics extensively, you may extract and paraphrase the balance sheet portion while preserving the exact meaning, language, and context
    4. **No interpretation:** Extract only what is explicitly stated, without adding inferences
    5. **Include both explicit and implicit guidance:** Capture both specific timelines/conditions and general data-dependent approaches

    **Output Format:**
    - List extracted statements, separated by new lines
    - If no relevant statements found, output a new line symbol. 
    - No explanations or additional text
"""

EMPLOYMENT_FILTERING_PROMPT = """ 
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand which factor is the central to the current discussion of the FED. (Employment or Inflation)

    Your task is to analyze a set of statements that are relevant to determine a sentiment of the FED and extract the statements that could be important according to the guidelines.

    Output only the statements that could be important according to the guidelines, separated by new lines.

    Guidelines:
    1. Employment:
        - When scanning the transcript, look for any of the following keywords or phrases related to employment: "labor market," "employment," "jobs," "unemployment," "job gains," "wage growth," "labor force participation," "maximum employment," "tight labor markets," "worker shortages," or "underemployment."
        Additionally, capture any qualitative references that reflect the Fed's assessment of the labor market:

        - Positive Employment Focus:
            - "Job growth remains strong/robust,"
            - "Unemployment at historically low levels,"
            - "Wages are rising,"
            - "We aim to support further employment gains."
        - Negative or Cautious Employment Focus:
            - "Signs of labor market softening,"
            - "Rising unemployment claims,"
            - "Concerns about layoffs or slowing wage growth,"
            - "Risk of higher unemployment."
        Make sure to extract the exact sentences or quotes that mention these points.

    2. Inflation:
        - Look for any mentions of inflation, price stability, or the 2% target. This may include: "inflation," "price stability," "price pressures," "inflation expectations," "cost pressures," "inflation target," "2%," "core inflation," "headline inflation."
        - Include qualitative assessments as well:
            - High or Concerning Inflation:
                - "Inflation is elevated/uncomfortably high,"
                - "Ongoing concerns about persistent price pressures,"
                - "Risk of unanchored inflation expectations."
            - Easing or Moderating Inflation:
                - "Inflation is moderating/coming down,"
                - "Price pressures have started to soften,"
                - "Reduced inflationary pressures due to supply-chain improvements."
            Again, extract the exact sentences or statements referencing inflation. If they discuss the underlying drivers (e.g., supply chain, wages, commodity prices) specifically in the context of inflation, include those as well.

    3. Explicit Prioritization:
        The Fed sometimes explicitly comments on balancing employment vs. inflation. Pay attention to any statements in which officials discuss weighing trade-offs or prioritizing one goal over the other:
        - Statements suggesting inflation is the primary focus:
            - "We must bring inflation down even if it impacts employment,"
            - "Our priority is price stability at this juncture."
        - Statements suggesting employment is the primary focus:
            - "We cannot jeopardize the labor market recovery,"
            - "We're concerned about higher unemployment if we tighten too quickly."
        Extract these verbatim (or with accurate paraphrasing). 

    Statements:
    {statements}

    Important statements:
"""

FORWARD_GUIDANCE_FILTERING_PROMPT = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand whether the FED is providing clear and explicit forward guidance or unclear and implicit forward guidance.

    Your task is to analyze a set of statements that are relevant to determine sentiment of the FED and extract the statements that could be important according to the guidelines.

    Output only the statements that could be important according to the guidelines, separated by new lines.

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

    Extract each statement verbatim (or with precise paraphrasing), separated by new lines. If provided transcript does not contain any relevant statements, do not output anything.

    Statements:
    {statements}

    Important statements:
"""

ECONOMIC_OUTLOOK_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    **Task:** Extract statements from FED press conference transcripts that are specifically relevant to general economic outlook and conditions, excluding labor market and inflation topics.

    **What to Extract:**
    - Complete, coherent statements about economic growth, GDP, and output conditions
    - References to output gap, economic slack, or capacity utilization
    - Assessments of overall economic health, momentum, and resilience
    - Views on productivity, business investment, and capital formation
    - Consumer spending patterns and household financial conditions
    - Economic projections and growth forecasts (excluding employment/inflation)
    - Financial conditions, credit markets, and lending standards
    - Recession risks, economic vulnerabilities, or downside risks
    - Questions that provide crucial context for economic outlook answers (include both question and answer)

    **Examples of Target Content:**
    - **Growth conditions:** "Economic growth remains robust," "GDP is expanding above trend," "Economic activity has slowed"
    - **Output gap:** "The economy is operating above potential," "Significant economic slack remains," "Output gap has narrowed"
    - **Economic resilience:** "The economy has shown remarkable resilience," "Economic fundamentals remain strong," "Vulnerabilities in the financial system"
    - **Business conditions:** "Business investment is picking up," "Corporate earnings are strong," "Credit conditions have tightened"
    - **Forward outlook:** "We expect continued economic expansion," "Risks to the economic outlook have increased"

    **Extraction Rules:**
    1. **Exclude labor/inflation:** Do not extract statements primarily about employment, jobs, unemployment, inflation, or price stability
    2. **Mixed topics:** If a statement covers economic outlook plus labor/inflation, extract only the general economic portion
    3. **Context preservation:** Ensure extracted statements are self-contained and understandable. Include necessary surrounding content if context is missing
    4. **Paraphrasing:** Only when a long statement mixes multiple topics extensively, you may extract and paraphrase the economic outlook portion while preserving the exact meaning, language, and context
    5. **Include projections:** Capture economic forecasts and projections that aren't specifically about employment or inflation
    6. **No interpretation:** Extract only what is explicitly stated, without adding inferences

    **Output Format:**
    - List extracted statements, separated by new lines
    - If no relevant statements found, output a new line symbol.
    - No explanations or additional text
"""

if __name__ == "__main__":
    data = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv")
    texts = data['sentiment_summary'].tolist()

    results =  asyncio.run(tqdm.gather(*[summarize_transcript(employment_prompt, text) for text in texts]))
    data['sentiment_summary'] = results
    data.to_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv", index=False)