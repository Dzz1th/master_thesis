import pandas as pd 
import asyncio
from tqdm.asyncio import tqdm   


sentiment_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that defines which statements in the transcript could be important to understand a central bank policy stance (hawkish vs dovish).

    Your task is to analyze a part of a transcript and extract the statements that could be important to asses a sentiment of the FED, whether it is hawkish or dovish.

    Please output only the statements that could be important according to the guidelines, separated by new lines. Do not use any other separators.
    Do not include any mentions of time, dates, year or FED Chairman's name. It is very important.

    Guidelines:
    1. Extract statements that could be important to assess what is the main focus of the FED - employment or inflation.
        When scanning the transcript, look for any of the following keywords or phrases related to employment: “labor market,” “employment,” “jobs,” “unemployment,” “job gains,” “wage growth,” “labor force participation,” “maximum employment,” “tight labor markets,” “worker shortages,” or “underemployment.”
        Additionally, capture any qualitative references that reflect the Fed’s assessment of the labor market:

        - Positive Employment Focus:
            - “Job growth remains strong/robust,”
            - “Unemployment at historically low levels,”
            - “Wages are rising,”
            - “We aim to support further employment gains.”
        - Negative or Cautious Employment Focus:
            - “Signs of labor market softening,”
            - “Rising unemployment claims,”
            - “Concerns about layoffs or slowing wage growth,”
            - “Risk of higher unemployment.”
        Make sure to extract the exact sentences or quotes that mention these points.

        Also Look for any mentions of inflation, price stability, or the 2% target. This may include: “inflation,” “price stability,” “price pressures,” “inflation expectations,” “cost pressures,” “inflation target,” “2%,” “core inflation,” “headline inflation.”
        - Include qualitative assessments as well:
            - High or Concerning Inflation:
                - “Inflation is elevated/uncomfortably high,”
                - “Ongoing concerns about persistent price pressures,”
                - “Risk of unanchored inflation expectations.”
            - Easing or Moderating Inflation:
                - “Inflation is moderating/coming down,”
                - “Price pressures have started to soften,”
                - “Reduced inflationary pressures due to supply-chain improvements.”
        Again, extract the exact sentences or statements referencing inflation. If they discuss the underlying drivers (e.g., supply chain, wages, commodity prices) specifically in the context of inflation, include those as well.

        The Fed sometimes explicitly comments on balancing employment vs. inflation. Pay attention to any statements in which officials discuss weighing trade-offs or prioritizing one goal over the other:
        - Statements suggesting inflation is the primary focus:
            - “We must bring inflation down even if it impacts employment,”
            - “Our priority is price stability at this juncture.”
        - Statements suggesting employment is the primary focus:
            - “We cannot jeopardize the labor market recovery,”
            - “We’re concerned about higher unemployment if we tighten too quickly.”
        Extract these verbatim (or with accurate paraphrasing). 

    2. Extract Language on Economic Conditions
        - Growth Outlook
            - Hawkish Clues: Emphasis on robust economic growth, strong demand, or economic expansion running faster than potential.
            - Dovish Clues: Concern about slowdown, below-trend growth, or risk of recession.

    3. Extract Language on Policy Trajectory / Forward Guidance
        - Interest Rate Path
            - Explicit Forward Guidance:
                - Look for any sentences mentioning a specific timeline or conditions under which the Fed will raise or lower interest rates. For example, “We do not expect to raise rates until at least next year,” “We will begin lowering rates once inflation has fallen to 2% for at least three months.”
            -Implicit Forward Guidance:
                - Identify statements about rates that are open-ended or conditional, but lack details about timing or thresholds. For example “We will adjust policy as data evolve,” “We remain data-dependent and will decide on a meeting-by-meeting basis.”
        - Balance Sheet Policy
            - Explicit Forward Guidance:
                - Watch for direct mentions of the pace or duration of quantitative tightening (QT) or expansions (QE). For example, “We will continue to reduce our holdings at $60 billion per month until mid-2025.”
            -Implicit Forward Guidance:
                - Note any references to the balance sheet that do not include a specific plan or timeframe. “We will monitor market conditions and adjust our balance sheet approach if necessary.”
        - Policy Lags and Caution
            - Hawkish Clues: Downplaying lags of monetary policy, indicating a willingness to keep tightening despite acknowledging some lagged effects.
            - Dovish Clues: Emphasis on the significant lags of monetary policy, implying caution or a wait-and-see approach before acting further.
        - Conditions / Thresholds for Policy Changes
            - Clear Conditions:
                - Identify any statements describing exact economic thresholds (e.g., unemployment rate, inflation rate) for policy shifts. For example, “We will maintain rates until inflation is on track to moderately exceed 2% for some time.”
            - Generic Conditions:
                - Flag statements that mention broad considerations but do not specify numeric targets or time frames. For example: “Our decisions will be guided by overall economic conditions and financial stability concerns.”
        - Strength of Language:
            - Highlight strong phrases indicating commitment to a future path (“We are determined to keep rates at current levels for the next few quarters”) versus hedged language (“We may consider pausing if growth slows, but it’s too soon to say”).
            - Collect any qualifiers that weaken clarity: “if needed,” “depending on the data,” “we remain flexible,” “we haven’t decided yet.”
            - Capture any unambiguous promises or timelines: “We will keep rates near zero until the labor market has fully recovered.”

    4. Risk Assessment and “Balance of Risks”
        - Risk to Growth vs. Risk to Inflation
            - Hawkish Clues: Focus on upside inflation risks as more critical than downside growth risks. Statements like “inflation is the bigger threat.”
            - Dovish Clues: Highlighting “uncertainty about economic momentum,” or a concern that overly tight policy could lead to recession, thus leaning toward caution in raising rates.
        - Global Factors and Financial Stability
            - Hawkish Clues: Dismissing or downplaying global economic uncertainties, focusing instead on domestic inflation control as the priority.
            - Dovish Clues: Emphasis on global economic headwinds, financial volatility, or other reasons to slow the pace of rate hikes.

    5. Press Conference Tone 
        - Chair’s Responses
            - Watch for Tone: Fed Chair remarks can be more candid during Q&A. Look for repetitive emphasis on “inflation fight is not over” (hawkish) vs. “seeing signs that we may have done enough” (dovish).
            - Clarifying Data Dependency: If the Chair repeatedly says policy depends on data but underscores that data still shows persistent inflation, that is more hawkish. If they note that inflation readings are improving, that is more dovish.
            - Specific Phrases to Note
                - Hawkish: “We will do whatever it takes,” “remain resolute,” “strong inflationary pressures,” “further tightening may be appropriate.”
                - Dovish: “We are prepared to adjust policy if risks emerge,” “policy is sufficiently restrictive,” “monitoring the need for additional accommodation or steady rates.”

    6. Summary of economics projections
        - Dot Plot
            - Hawkish Clues: An upward shift in the median dot for upcoming years, indicating higher rates or a longer holding period before cutting.
            - Dovish Clues: A lower or flat trajectory of the dots, or a signal that rate hikes may come to an end sooner than previously projected.
        - GDP, Unemployment, and Inflation Projections
            - Hawkish Clues: Rising inflation forecasts, or forecasts of stronger growth (meaning the Fed sees more inflation pressure).
            - Dovish Clues: Lower inflation forecasts, higher unemployment forecasts, or slower growth forecasts pointing to caution in further tightening.

    Extract all the statements verbatim (or with precise paraphrasing). If provided transcript does not contain any relevant statements, do not output anything.
"""

employment_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    Your task is to analyze a part of a FED press conference transcript and extract only the statements relevant to the labor market and employment.

    **Guidelines:**
    - Look for sentences that contain any of the following keywords or phrases:  
      **“labor market,” “employment,” “jobs,” “unemployment,” “job gains,” “wage growth,” “labor force participation,” “maximum employment,” “tight labor markets,” “worker shortages,” or “underemployment.”**
    - Extract full sentences that contain qualitative assessments about employment:  
      - **Positive employment indicators**:  
        - “Job growth remains strong/robust.”  
        - “Unemployment at historically low levels.”  
        - “Wages are rising.”  
        - “We aim to support further employment gains.”  
      - **Negative or cautious employment indicators**:  
        - “Signs of labor market softening.”  
        - “Rising unemployment claims.”  
        - “Concerns about layoffs or slowing wage growth.”  
        - “Risk of higher unemployment.”  

    - If a statement discusses **both employment and inflation in the same sentence**, extract only the employment-related portion.  
    - If it is difficult to extract only the relevant part from a long sentence, **you may paraphrase, but the meaning must remain exactly the same**.  
    - **Do not generate new interpretations, inferences, or additional information beyond what is explicitly stated.**  
    - If no relevant employment statements are found in the chunk, output **an empty line or a single dot (`"."`)**.  

    **Output only the extracted statements, separated by new lines. Do not include explanations or additional text.**
"""

inflation_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    Your task is to analyze a part of a FED press conference transcript and extract only the statements relevant to inflation and price stability.

    **Guidelines:**
    - Look for sentences that contain any of the following keywords or phrases:  
      **“inflation,” “price stability,” “price pressures,” “inflation expectations,” “cost pressures,” “inflation target,” “2%,” “core inflation,” “headline inflation.”**
    - Extract full sentences that contain qualitative assessments about inflation:  
      - **High or concerning inflation**:  
        - “Inflation is elevated/uncomfortably high.”  
        - “Ongoing concerns about persistent price pressures.”  
        - “Risk of unanchored inflation expectations.”  
      - **Easing or moderating inflation**:  
        - “Inflation is moderating/coming down.”  
        - “Price pressures have started to soften.”  
        - “Reduced inflationary pressures due to supply-chain improvements.”  

    - If a statement discusses **both inflation and employment in the same sentence**, extract only the inflation-related portion.  
    - If it is difficult to extract only the relevant part from a long sentence, **you may paraphrase, but the meaning must remain exactly the same**.  
    - **Do not generate new interpretations, inferences, or additional information beyond what is explicitly stated.**  
    - If no relevant inflation statements are found in the chunk, output **an empty line or a single dot (`"."`)**.  

    **Output only the extracted statements, separated by new lines. Do not include explanations or additional text.**
"""




forward_guidance_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand whether the FED is providing clear and explicit forward guidance or unclear and implicit forward guidance.

    Your task is to analyze a part of a transcript and extract the statements that could be important according to the guidelines.

    Output only the statements that could be important according to the guidelines, separated by new lines.

    Guidelines:
    1. Interest Rate Path
        - Explicit Forward Guidance:
            - Look for any sentences mentioning a specific timeline or conditions under which the Fed will raise or lower interest rates. For example, “We do not expect to raise rates until at least next year,” “We will begin lowering rates once inflation has fallen to 2% for at least three months.”
        -Implicit Forward Guidance:
            - Identify statements about rates that are open-ended or conditional, but lack details about timing or thresholds. For example “We will adjust policy as data evolve,” “We remain data-dependent and will decide on a meeting-by-meeting basis.”
    2. Balance Sheet Policy
        - Explicit Forward Guidance:
            - Watch for direct mentions of the pace or duration of quantitative tightening (QT) or expansions (QE). For example, “We will continue to reduce our holdings at $60 billion per month until mid-2025.”
        -Implicit Forward Guidance:
            - Note any references to the balance sheet that do not include a specific plan or timeframe. “We will monitor market conditions and adjust our balance sheet approach if necessary.”
    3. Conditions / Thresholds for Policy Changes
        - Clear Conditions:
            - Identify any statements describing exact economic thresholds (e.g., unemployment rate, inflation rate) for policy shifts. For example, “We will maintain rates until inflation is on track to moderately exceed 2% for some time.”
        - Generic Conditions:
            - Flag statements that mention broad considerations but do not specify numeric targets or time frames. For example: “Our decisions will be guided by overall economic conditions and financial stability concerns.”
    4. Tone in Q&A or Additional Remarks
        - Strength of Language:
            - Highlight strong phrases indicating commitment to a future path (“We are determined to keep rates at current levels for the next few quarters”) versus hedged language (“We may consider pausing if growth slows, but it’s too soon to say”).
            - Collect any qualifiers that weaken clarity: “if needed,” “depending on the data,” “we remain flexible,” “we haven’t decided yet.”
            - Capture any unambiguous promises or timelines: “We will keep rates near zero until the labor market has fully recovered.”

    Extract each statement verbatim (or with precise paraphrasing). If provided transcript does not contain any relevant statements, do not output anything.
"""

interest_rate_prompt = """
    You are an international expert in macroeconomics and central bank policy.

    You are provided with guidelines to extract statements from a transcript that reveal the FED's perspective on the future path of interest rates.

    Your task is to analyze a section of the transcript and extract only the statements that are relevant to the interest rate trajectory, including both explicit and implicit forward guidance.

    Output only the statements that are important according to these guidelines, each on a new line. Do not include any additional text or separators.

    Guidelines:
    1. Explicit Forward Guidance:
        - Identify sentences that state specific timelines or conditions for future interest rate moves. For example: "We do not expect to raise rates until at least next year" or "We will begin lowering rates once inflation falls to 2% for at least three months."
    2. Implicit Forward Guidance:
        - Extract statements that refer to a data-dependent or open-ended approach regarding interest rate adjustments. Examples include: "We will adjust policy as data evolve" or "Our decisions on rates will be made on a meeting-by-meeting basis."
    3. Directional and Qualitative Clues:
        - Look for language indicating the direction of rate moves, such as "tightening," "raising," "increasing," "lowering," "easing," or "pausing."
    4. Conditions and Economic Triggers:
        - Capture any references to economic indicators (e.g., inflation, employment) that are mentioned as triggers for rate changes.
    
    Extract each relevant statement verbatim (or with precise paraphrasing). If no relevant statements are found, DO NOT OUTPUT ANYTHING.
"""

balance_sheet_prompt = """
    You are an international expert in macroeconomics and central bank policy.

    You are provided with guidelines to extract statements from a transcript that reveal the FED's strategy regarding its balance sheet trajectory, including policies related to asset purchases (QE) and quantitative tightening (QT).

    Your task is to analyze a section of the transcript and extract only the statements that are relevant to balance sheet policy.

    Output only the statements that are important according to these guidelines, each on a new line. Do not include any additional text or separators.

    Guidelines:
    1. Asset Purchases (QE):
        - Identify any statements that mention the expansion of the balance sheet through asset purchases. Look for explicit mentions such as "We will continue asset purchases at [specific amount]" or cues that indicate an ongoing or expanded QE program.
    2. Quantitative Tightening (QT):
        - Extract sentences that refer to the reduction of the balance sheet, such as "We will reduce our holdings at [specific pace or timeline]" or phrases that indicate winding down asset purchases.
    3. Explicit vs. Implicit Guidance:
        - For both QE and QT, capture statements offering explicit details (amounts, timelines) as well as conditional or implicit statements like "We will monitor market conditions and adjust our balance sheet policy if necessary."
    4. Contextual Triggers:
        - Include any references linking balance sheet decisions to economic conditions, such as inflation or growth indicators.
    
    Extract each relevant statement verbatim (or with precise paraphrasing). If no relevant statements are found, DO NOT OUTPUT ANYTHING. 
"""

employment_filtering_prompt = """ 
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand which factor is the central to the current discussion of the FED. (Employment or Inflation)

    Your task is to analyze a set of statements that are relevant to determine a sentiment of the FED and extract the statements that could be important according to the guidelines.

    Output only the statements that could be important according to the guidelines, separated by new lines.

    Guidelines:
    1. Employment:
        - When scanning the transcript, look for any of the following keywords or phrases related to employment: “labor market,” “employment,” “jobs,” “unemployment,” “job gains,” “wage growth,” “labor force participation,” “maximum employment,” “tight labor markets,” “worker shortages,” or “underemployment.”
        Additionally, capture any qualitative references that reflect the Fed’s assessment of the labor market:

        - Positive Employment Focus:
            - “Job growth remains strong/robust,”
            - “Unemployment at historically low levels,”
            - “Wages are rising,”
            - “We aim to support further employment gains.”
        - Negative or Cautious Employment Focus:
            - “Signs of labor market softening,”
            - “Rising unemployment claims,”
            - “Concerns about layoffs or slowing wage growth,”
            - “Risk of higher unemployment.”
        Make sure to extract the exact sentences or quotes that mention these points.

    2. Inflation:
        - Look for any mentions of inflation, price stability, or the 2% target. This may include: “inflation,” “price stability,” “price pressures,” “inflation expectations,” “cost pressures,” “inflation target,” “2%,” “core inflation,” “headline inflation.”
        - Include qualitative assessments as well:
            - High or Concerning Inflation:
                - “Inflation is elevated/uncomfortably high,”
                - “Ongoing concerns about persistent price pressures,”
                - “Risk of unanchored inflation expectations.”
            - Easing or Moderating Inflation:
                - “Inflation is moderating/coming down,”
                - “Price pressures have started to soften,”
                - “Reduced inflationary pressures due to supply-chain improvements.”
            Again, extract the exact sentences or statements referencing inflation. If they discuss the underlying drivers (e.g., supply chain, wages, commodity prices) specifically in the context of inflation, include those as well.

    3. Explicit Prioritization:
        The Fed sometimes explicitly comments on balancing employment vs. inflation. Pay attention to any statements in which officials discuss weighing trade-offs or prioritizing one goal over the other:
        - Statements suggesting inflation is the primary focus:
            - “We must bring inflation down even if it impacts employment,”
            - “Our priority is price stability at this juncture.”
        - Statements suggesting employment is the primary focus:
            - “We cannot jeopardize the labor market recovery,”
            - “We’re concerned about higher unemployment if we tighten too quickly.”
        Extract these verbatim (or with accurate paraphrasing). 

    Statements:
    {statements}

    Important statements:
"""



forward_guidance_filtering_prompt = """
    You are an international expert in macroeconomics and central bank policy. 

    You are given a guidelines that can help you understand which statements in the transcript could be important to understand whether the FED is providing clear and explicit forward guidance or unclear and implicit forward guidance.

    Your task is to analyze a set of statements that are relevant to determine sentiment of the FED and extract the statements that could be important according to the guidelines.

    Output only the statements that could be important according to the guidelines, separated by new lines.

    Guidelines:
    1. Interest Rate Path
        - Explicit Forward Guidance:
            - Look for any sentences mentioning a specific timeline or conditions under which the Fed will raise or lower interest rates. For example, “We do not expect to raise rates until at least next year,” “We will begin lowering rates once inflation has fallen to 2% for at least three months.”
        -Implicit Forward Guidance:
            - Identify statements about rates that are open-ended or conditional, but lack details about timing or thresholds. For example “We will adjust policy as data evolve,” “We remain data-dependent and will decide on a meeting-by-meeting basis.”
    2. Balance Sheet Policy
        - Explicit Forward Guidance:
            - Watch for direct mentions of the pace or duration of quantitative tightening (QT) or expansions (QE). For example, “We will continue to reduce our holdings at $60 billion per month until mid-2025.”
        -Implicit Forward Guidance:
            - Note any references to the balance sheet that do not include a specific plan or timeframe. “We will monitor market conditions and adjust our balance sheet approach if necessary.”
    3. Conditions / Thresholds for Policy Changes
        - Clear Conditions:
            - Identify any statements describing exact economic thresholds (e.g., unemployment rate, inflation rate) for policy shifts. For example, “We will maintain rates until inflation is on track to moderately exceed 2% for some time.”
        - Generic Conditions:
            - Flag statements that mention broad considerations but do not specify numeric targets or time frames. For example: “Our decisions will be guided by overall economic conditions and financial stability concerns.”
    4. Tone in Q&A or Additional Remarks
        - Strength of Language:
            - Highlight strong phrases indicating commitment to a future path (“We are determined to keep rates at current levels for the next few quarters”) versus hedged language (“We may consider pausing if growth slows, but it’s too soon to say”).
            - Collect any qualifiers that weaken clarity: “if needed,” “depending on the data,” “we remain flexible,” “we haven’t decided yet.”
            - Capture any unambiguous promises or timelines: “We will keep rates near zero until the labor market has fully recovered.”

    Extract each statement verbatim (or with precise paraphrasing), separated by new lines. If provided transcript does not contain any relevant statements, do not output anything.

    Statements:
    {statements}

    Important statements:
"""




if __name__ == "__main__":
    data = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv")
    texts = data['sentiment_summary'].tolist()

    results =  asyncio.run(tqdm.gather(*[summarize_transcript(employment_prompt, text) for text in texts]))
    data['sentiment_summary'] = results
    data.to_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv", index=False)